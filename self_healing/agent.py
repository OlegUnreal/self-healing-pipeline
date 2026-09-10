"""The repair loop: observe → propose → apply → verify, up to max_attempts.

A second loop, `heal_with_tools`, lets the model call the sandboxed Python
tool registry instead of emitting a raw diff in one shot.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from . import logging as logmod
from .patcher import apply_diff
from .sandbox import run_code  # noqa: F401
from .tools import ToolRegistry, Workspace, default_registry
from .tools.base import ToolResult
from .verifier import verify

log = logmod.get_logger(__name__)


@dataclass
class Attempt:
    n: int
    diff: str
    passed: bool
    output: str
    error: str = ""
    diff_hash: str = ""
    tools: list[str] = field(default_factory=list)


@dataclass
class HealReport:
    history: list[Attempt] = field(default_factory=list)
    success: bool = False
    final_source: str = ""
    events: logmod.EventLog = field(default_factory=logmod.EventLog)
    error: str = ""
    tool_trace: list[dict[str, Any]] = field(default_factory=list)

    @property
    def attempts(self) -> int:
        return len(self.history)

    def __iter__(self) -> Iterable[Attempt]:
        return iter(self.history)


def heal(
    source: Path,
    test: str,
    propose_diff,
    max_attempts: int = 5,
    logger: logging.Logger | None = None,
    cwd: str | Path | None = None,
) -> HealReport:
    lg = logger or log
    report = HealReport()
    workdir = Path(cwd) if cwd is not None else source.parent
    for n in range(1, max_attempts + 1):
        with logmod.attempt_span(lg, n, source=str(source)):
            try:
                result = verify(test, cwd=workdir)
            except Exception as e:  # noqa: BLE001
                msg = f"verifier crashed: {e}"
                lg.exception("verify.crash", attempt=n)
                report.history.append(Attempt(n, "", False, "", error=msg))
                report.events.add("verify_crash", attempt=n, error=str(e))
                report.error = msg
                continue

            if result.exit_code == 0:
                report.history.append(Attempt(n, "", True, result.stdout, diff_hash=""))
                report.success = True
                report.final_source = source.read_text(encoding="utf-8")
                report.events.add("success", attempt=n)
                lg.info("heal.success", attempt=n)
                return report

            tb = result.stderr or result.stdout
            try:
                diff = propose_diff(tb) or ""
            except Exception as e:  # noqa: BLE001
                msg = f"proposer failed: {e}"
                lg.exception("propose.crash", attempt=n)
                report.history.append(Attempt(n, "", False, tb, error=msg))
                report.events.add("propose_crash", attempt=n, error=str(e))
                report.error = msg
                continue

            if not diff.strip():
                report.history.append(Attempt(n, "", False, tb, error="empty diff", diff_hash=""))
                report.events.add("empty_diff", attempt=n)
                lg.warning("propose.empty", attempt=n)
                continue

            try:
                ok = apply_diff(source, diff)
            except Exception as e:  # noqa: BLE001
                msg = f"patcher crashed: {e}"
                lg.exception("apply.crash", attempt=n)
                report.history.append(Attempt(n, diff, False, tb, error=msg, diff_hash=logmod._hash(diff)))
                report.events.add("apply_crash", attempt=n, error=str(e))
                report.error = msg
                continue

            report.history.append(Attempt(n, diff, ok, tb, diff_hash=logmod._hash(diff)))
            report.events.add("applied", attempt=n, ok=ok, diff_hash=logmod._hash(diff))
            lg.info("apply.result", attempt=n, ok=ok)

    report.final_source = source.read_text(encoding="utf-8")
    if not report.success:
        report.error = report.error or "exhausted attempts without green tests"
        lg.error("heal.exhausted", attempts=report.attempts, error=report.error)
    return report


Planner = Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]]


def heal_with_tools(
    source: Path,
    test: str,
    planner: Planner,
    max_steps: int = 12,
    registry: ToolRegistry | None = None,
    workspace: Workspace | None = None,
    logger: logging.Logger | None = None,
) -> HealReport:
    lg = logger or log
    tools = registry or default_registry()
    ws = workspace or Workspace(source.parent)
    report = HealReport()
    schemas = tools.schemas()
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You repair a failing Python workspace. Use tools to inspect, "
                "patch, compile and re-test. Prefer apply_patch over write_file. "
                "Stop when run_tests reports passed=true."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Workspace root: {ws.root}\n"
                f"Primary file: {source.name}\n"
                f"Test to make green:\n{test}"
            ),
        },
    ]

    used: list[str] = []
    last_output = ""
    for n in range(1, max_steps + 1):
        with logmod.attempt_span(lg, n, source=str(source), mode="tools"):
            try:
                decision = planner(messages, schemas) or {}
            except Exception as e:  # noqa: BLE001
                msg = f"planner failed: {e}"
                lg.exception("planner.crash", step=n)
                report.history.append(Attempt(n, "", False, last_output, error=msg, tools=list(used)))
                report.events.add("planner_crash", attempt=n, error=str(e))
                report.error = msg
                continue

            content = decision.get("content") or ""
            calls = decision.get("tool_calls") or []
            if content:
                messages.append({"role": "assistant", "content": content})

            if not calls:
                result = verify(test, cwd=ws.root)
                last_output = result.stderr or result.stdout
                passed = result.exit_code == 0
                report.history.append(Attempt(n, content, passed, last_output, tools=list(used)))
                if passed:
                    report.success = True
                    report.final_source = source.read_text(encoding="utf-8")
                    report.events.add("success", attempt=n)
                    return report
                report.events.add("no_tool_calls", attempt=n)
                continue

            for call in calls:
                name = call.get("name") or ""
                raw_args = call.get("arguments") or {}
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args) if raw_args else {}
                    except json.JSONDecodeError:
                        raw_args = {}
                if name == "finish":
                    result = verify(test, cwd=ws.root)
                    last_output = result.stderr or result.stdout
                    passed = result.exit_code == 0
                    report.history.append(Attempt(n, "", passed, last_output, tools=list(used) + ["finish"]))
                    if passed:
                        report.success = True
                        report.final_source = source.read_text(encoding="utf-8")
                        report.events.add("success", attempt=n)
                        return report
                    continue

                result = tools.call(ws, name, raw_args)
                used.append(name)
                report.tool_trace.append({"step": n, "tool": name, "ok": result.ok, "error": result.error})
                report.events.add("tool_call", attempt=n, tool=name, ok=result.ok)
                messages.append({"role": "tool", "name": name, "content": result.to_llm()})
                lg.info("tool.result", step=n, tool=name, ok=result.ok)

            result = verify(test, cwd=ws.root)
            last_output = result.stderr or result.stdout
            passed = result.exit_code == 0
            report.history.append(Attempt(n, "", passed, last_output, tools=list(used)))
            if passed:
                report.success = True
                report.final_source = source.read_text(encoding="utf-8")
                report.events.add("success", attempt=n)
                return report

    report.final_source = source.read_text(encoding="utf-8") if source.exists() else ""
    if not report.success:
        report.error = report.error or "exhausted tool steps without green tests"
    return report
