"""The repair loop: observe → propose → apply → verify, up to max_attempts."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from . import logging as logmod
from .patcher import apply_diff
from .sandbox import run_code
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


@dataclass
class HealReport:
    history: list[Attempt] = field(default_factory=list)
    success: bool = False
    final_source: str = ""
    events: logmod.EventLog = field(default_factory=logmod.EventLog)
    error: str = ""

    @property
    def attempts(self) -> int:
        return len(self.history)


def heal(
    source: Path,
    test: str,
    propose_diff,
    max_attempts: int = 5,
    logger: logging.Logger | None = None,
) -> HealReport:
    """`propose_diff(traceback) -> str` is the LLM hook. Kept injectable for tests.

    Empty or rejected diffs are skipped (counted as a failed attempt) so a
    flaky model cannot burn the whole budget on one bad response. Every
    failure mode — LLM exception, empty diff, rejected patch, verify crash —
    is caught and recorded instead of aborting the loop.
    """
    lg = logger or log
    report = HealReport()
    for n in range(1, max_attempts + 1):
        with logmod.attempt_span(lg, n, source=str(source)):
            try:
                result = verify(test)
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
                report.history.append(
                    Attempt(n, "", False, tb, error="empty diff", diff_hash="")
                )
                report.events.add("empty_diff", attempt=n)
                lg.warning("propose.empty", attempt=n)
                continue

            try:
                ok = apply_diff(source, diff)
            except Exception as e:  # noqa: BLE001
                msg = f"patcher crashed: {e}"
                lg.exception("apply.crash", attempt=n)
                report.history.append(
                    Attempt(n, diff, False, tb, error=msg, diff_hash=logmod._hash(diff))
                )
                report.events.add("apply_crash", attempt=n, error=str(e))
                report.error = msg
                continue

            report.history.append(
                Attempt(n, diff, ok, tb, diff_hash=logmod._hash(diff))
            )
            report.events.add("applied", attempt=n, ok=ok, diff_hash=logmod._hash(diff))
            lg.info("apply.result", attempt=n, ok=ok)

    report.final_source = source.read_text(encoding="utf-8")
    if not report.success:
        report.error = report.error or "exhausted attempts without green tests"
        lg.error("heal.exhausted", attempts=report.attempts, error=report.error)
    return report
