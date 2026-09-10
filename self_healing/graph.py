"""Graph orchestrator for the tool-calling repair loop.

The capability boundary stays in `tools/` + sandbox. This module only
routes Observe → Plan → Tools → Observe with an explicit budget.

LangGraph is an optional extra. When it is installed, `build_heal_graph()`
compiles a real `StateGraph`. When it is not, `heal_with_graph()` runs the
same nodes through a local interpreter so CI and `--demo-graph` never
depend on the extra.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

from .agent import Attempt, HealReport, Planner
from . import logging as logmod
from .tools import ToolRegistry, Workspace, default_registry
from .verifier import verify

log = logmod.get_logger(__name__)

END = "__end__"
MUTATING_TOOLS = frozenset({"write_file", "apply_patch"})

MutationApprover = Callable[[dict[str, Any]], bool]


def langgraph_available() -> bool:
    try:
        import langgraph  # noqa: F401
        from langgraph.graph import StateGraph  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class GraphContext:
    source: Path
    test: str
    planner: Planner
    workspace: Workspace
    registry: ToolRegistry
    approve_mutations: bool = False
    mutation_approver: MutationApprover | None = None
    logger: Any = None


def initial_state(source: Path, test: str, max_steps: int) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You repair a failing Python workspace. Use tools to inspect, "
                    "patch, compile and re-test. Prefer apply_patch over write_file. "
                    "Stop when tests are green."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Primary file: {source.name}\n"
                    f"Test to make green:\n{test}"
                ),
            },
        ],
        "source_name": source.name,
        "test_code": test,
        "failure_class": "",
        "last_output": "",
        "steps": 0,
        "max_steps": max_steps,
        "passed": False,
        "error": "",
        "tool_trace": [],
        "pending_calls": [],
        "used_tools": [],
        "last_content": "",
    }


def observe_node(ctx: GraphContext, state: dict[str, Any]) -> dict[str, Any]:
    result = verify(state["test_code"], cwd=ctx.workspace.root)
    output = result.stderr or result.stdout
    passed = result.exit_code == 0
    kind = result.error or ("ok" if passed else "runtime_error")
    return {
        "passed": passed,
        "last_output": output,
        "failure_class": kind,
    }


def route_after_observe(state: dict[str, Any]) -> str:
    if state.get("passed"):
        return END
    if int(state.get("steps", 0)) >= int(state.get("max_steps", 0)):
        return "escalate"
    return "plan"


def plan_node(ctx: GraphContext, state: dict[str, Any]) -> dict[str, Any]:
    lg = ctx.logger or log
    steps = int(state.get("steps", 0)) + 1
    messages = list(state.get("messages") or [])
    try:
        decision = ctx.planner(messages, ctx.registry.schemas()) or {}
    except Exception as e:  # noqa: BLE001
        lg.exception("graph.planner_crash", step=steps)
        return {
            "steps": steps,
            "error": f"planner failed: {e}",
            "pending_calls": [],
            "last_content": "",
        }
    content = decision.get("content") or ""
    calls = list(decision.get("tool_calls") or [])
    if content:
        messages.append({"role": "assistant", "content": content})
    return {
        "steps": steps,
        "messages": messages,
        "pending_calls": calls,
        "last_content": content,
        "error": "",
    }


def route_after_plan(state: dict[str, Any]) -> str:
    if state.get("error", "").startswith("planner failed"):
        return "escalate"
    if state.get("pending_calls"):
        return "tools"
    return "observe"


def _approved(ctx: GraphContext, call: dict[str, Any]) -> bool:
    name = call.get("name") or ""
    if name not in MUTATING_TOOLS or not ctx.approve_mutations:
        return True
    if ctx.mutation_approver is None:
        return False
    try:
        return bool(ctx.mutation_approver(call))
    except Exception:  # noqa: BLE001
        return False


def tools_node(ctx: GraphContext, state: dict[str, Any]) -> dict[str, Any]:
    lg = ctx.logger or log
    messages = list(state.get("messages") or [])
    used = list(state.get("used_tools") or [])
    trace = list(state.get("tool_trace") or [])
    step = int(state.get("steps", 0))
    calls = list(state.get("pending_calls") or [])

    for call in calls:
        name = call.get("name") or ""
        raw_args = call.get("arguments") or {}
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                raw_args = {}

        if name == "finish":
            used.append("finish")
            trace.append({"step": step, "tool": "finish", "ok": True, "error": ""})
            messages.append({"role": "tool", "name": "finish", "content": "{\"ok\": true}"})
            continue

        if not _approved(ctx, call):
            used.append(name)
            err = f"mutation rejected: {name}"
            trace.append({"step": step, "tool": name, "ok": False, "error": err})
            messages.append({"role": "tool", "name": name, "content": json.dumps({"ok": False, "error": err})})
            lg.warning("graph.mutation_rejected", tool=name, step=step)
            continue

        result = ctx.registry.call(ctx.workspace, name, raw_args)
        used.append(name)
        trace.append({"step": step, "tool": name, "ok": result.ok, "error": result.error})
        messages.append({"role": "tool", "name": name, "content": result.to_llm()})
        lg.info("graph.tool", step=step, tool=name, ok=result.ok)

    return {
        "messages": messages,
        "used_tools": used,
        "tool_trace": trace,
        "pending_calls": [],
    }


def escalate_node(ctx: GraphContext, state: dict[str, Any]) -> dict[str, Any]:
    err = state.get("error") or "exhausted tool steps without green tests"
    return {"error": err, "passed": False}


def report_from_state(ctx: GraphContext, state: dict[str, Any]) -> HealReport:
    report = HealReport()
    report.success = bool(state.get("passed"))
    report.error = "" if report.success else (state.get("error") or "")
    report.tool_trace = list(state.get("tool_trace") or [])
    src = ctx.source
    report.final_source = src.read_text(encoding="utf-8") if src.exists() else ""
    used = list(state.get("used_tools") or [])
    n = int(state.get("steps") or 0) or 1
    report.history.append(
        Attempt(
            n=n,
            diff=state.get("last_content") or "",
            passed=report.success,
            output=state.get("last_output") or "",
            error=report.error,
            tools=used,
        )
    )
    report.events.add("graph_done", attempt=n, success=report.success)
    return report


def iter_heal_graph(
    ctx: GraphContext,
    state: dict[str, Any] | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (node_name, state) after each node. Local interpreter."""
    current = dict(state or initial_state(ctx.source, ctx.test, 12))
    node: str = "observe"
    seen = 0
    while node and node != END and seen < 64:
        seen += 1
        if node == "observe":
            current.update(observe_node(ctx, current))
        elif node == "plan":
            current.update(plan_node(ctx, current))
        elif node == "tools":
            current.update(tools_node(ctx, current))
        elif node == "escalate":
            current.update(escalate_node(ctx, current))
        else:
            break
        yield node, dict(current)
        if node == "observe":
            node = route_after_observe(current)
        elif node == "plan":
            node = route_after_plan(current)
        elif node == "tools":
            node = "observe"
        elif node == "escalate":
            node = END
        else:
            node = END


def heal_with_graph(
    source: Path,
    test: str,
    planner: Planner,
    max_steps: int = 12,
    registry: ToolRegistry | None = None,
    workspace: Workspace | None = None,
    approve_mutations: bool = False,
    mutation_approver: MutationApprover | None = None,
    use_langgraph: bool = False,
    checkpointer: Any = None,
    thread_id: str = "heal",
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> HealReport:
    ctx = GraphContext(
        source=source,
        test=test,
        planner=planner,
        workspace=workspace or Workspace(source.parent),
        registry=registry or default_registry(),
        approve_mutations=approve_mutations,
        mutation_approver=mutation_approver,
    )
    start = initial_state(source, test, max_steps)

    if use_langgraph and langgraph_available():
        return _heal_with_langgraph(ctx, start, checkpointer, thread_id, on_event)

    last: dict[str, Any] = start
    for name, snap in iter_heal_graph(ctx, start):
        last = snap
        if on_event:
            on_event(name, snap)
    return report_from_state(ctx, last)


def build_heal_graph(ctx: GraphContext, checkpointer: Any = None):
    """Compile a LangGraph StateGraph. Raises ImportError if extra missing."""
    from langgraph.graph import END as LG_END
    from langgraph.graph import START, StateGraph

    def observe(state: dict[str, Any]) -> dict[str, Any]:
        return observe_node(ctx, state)

    def plan(state: dict[str, Any]) -> dict[str, Any]:
        return plan_node(ctx, state)

    def tools(state: dict[str, Any]) -> dict[str, Any]:
        return tools_node(ctx, state)

    def escalate(state: dict[str, Any]) -> dict[str, Any]:
        return escalate_node(ctx, state)

    def after_observe(state: dict[str, Any]) -> Literal["plan", "escalate", "__end__"]:
        nxt = route_after_observe(state)
        return LG_END if nxt == END else nxt  # type: ignore[return-value]

    def after_plan(state: dict[str, Any]) -> Literal["tools", "observe", "escalate"]:
        return route_after_plan(state)  # type: ignore[return-value]

    builder = StateGraph(dict)
    builder.add_node("observe", observe)
    builder.add_node("plan", plan)
    builder.add_node("tools", tools)
    builder.add_node("escalate", escalate)
    builder.add_edge(START, "observe")
    builder.add_conditional_edges(
        "observe",
        after_observe,
        {"plan": "plan", "escalate": "escalate", LG_END: LG_END},
    )
    builder.add_conditional_edges(
        "plan",
        after_plan,
        {"tools": "tools", "observe": "observe", "escalate": "escalate"},
    )
    builder.add_edge("tools", "observe")
    builder.add_edge("escalate", LG_END)
    kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    return builder.compile(**kwargs)


def _heal_with_langgraph(
    ctx: GraphContext,
    start: dict[str, Any],
    checkpointer: Any,
    thread_id: str,
    on_event: Callable[[str, dict[str, Any]], None] | None,
) -> HealReport:
    graph = build_heal_graph(ctx, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    last = start
    try:
        stream = graph.stream(start, config=config, stream_mode="updates")
        for update in stream:
            if not isinstance(update, dict):
                continue
            for node_name, payload in update.items():
                if isinstance(payload, dict):
                    last.update(payload)
                if on_event:
                    on_event(str(node_name), dict(last))
    except Exception:
        last = graph.invoke(start, config=config)
        if on_event:
            on_event("invoke", dict(last))
    return report_from_state(ctx, last)


def make_checkpointer(path: str | Path | None = None) -> Any:
    """Best-effort checkpointer.

    * no path + LangGraph → InMemorySaver
    * path + LangGraph sqlite extra → SqliteSaver
    * otherwise → None (local interpreter does not need one)
    """
    if path:
        try:
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver

            conn = sqlite3.connect(str(path), check_same_thread=False)
            return SqliteSaver(conn)
        except Exception:  # noqa: BLE001
            log.warning("graph.sqlite_unavailable", path=str(path))
            return None
    try:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()
    except Exception:  # noqa: BLE001
        try:
            from langgraph.checkpoint.memory import MemorySaver

            return MemorySaver()
        except Exception:  # noqa: BLE001
            return None
