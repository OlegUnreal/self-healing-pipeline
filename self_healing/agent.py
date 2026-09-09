"""The repair loop: observe → propose → apply → verify, up to max_attempts."""
from __future__ import annotations

from dataclasses import dataclass
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


def heal(source: Path, test: str, propose_diff, max_attempts: int = 5) -> list[Attempt]:
    """`propose_diff(traceback) -> str` is the LLM hook. Kept injectable for tests."""
    history: list[Attempt] = []
    for n in range(1, max_attempts + 1):
        result = verify(test)
        if result.exit_code == 0:
            history.append(Attempt(n, "", True, result.stdout))
            break
        diff = propose_diff(result.stderr or result.stdout)
        ok = apply_diff(source, diff)
        history.append(Attempt(n, diff, ok, result.stderr))
        if not ok:
            continue
    return history
