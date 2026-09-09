"""The repair loop: observe → propose → apply → verify, up to max_attempts."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .patcher import apply_diff
from .sandbox import run_code
from .verifier import verify


@dataclass
class Attempt:
    n: int
    diff: str
    passed: bool
    output: str
    error: str = ""


@dataclass
class HealReport:
    history: list[Attempt] = field(default_factory=list)
    success: bool = False
    final_source: str = ""

    @property
    def attempts(self) -> int:
        return len(self.history)


def heal(
    source: Path,
    test: str,
    propose_diff,
    max_attempts: int = 5,
) -> HealReport:
    """`propose_diff(traceback) -> str` is the LLM hook. Kept injectable for tests.

    Empty or rejected diffs are skipped (counted as a failed attempt) so a
    flaky model cannot burn the whole budget on one bad response.
    """
    history: list[Attempt] = []
    for n in range(1, max_attempts + 1):
        result = verify(test)
        if result.exit_code == 0:
            history.append(Attempt(n, "", True, result.stdout))
            return HealReport(history, True, source.read_text(encoding="utf-8"))

        try:
            diff = propose_diff(result.stderr or result.stdout) or ""
        except Exception as e:  # noqa: BLE001 - surface LLM failures, keep looping
            history.append(Attempt(n, "", False, result.stderr, error=str(e)))
            continue

        if not diff.strip():
            history.append(Attempt(n, "", False, result.stderr, error="empty diff"))
            continue

        ok = apply_diff(source, diff)
        history.append(Attempt(n, diff, ok, result.stderr))

    return HealReport(history, False, source.read_text(encoding="utf-8"))
