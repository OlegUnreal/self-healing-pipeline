"""Verifier: re-runs the test and decides pass/fail."""
from __future__ import annotations

from .sandbox import RunResult, run_code


def verify(test_code: str, timeout: float = 5.0) -> RunResult:
    """Run `test_code`; success == exit code 0 and no 'FAIL' in output."""
    result = run_code(test_code, timeout=timeout)
    passed = result.exit_code == 0 and "FAIL" not in result.stdout
    return RunResult(result.stdout, result.stderr, result.exit_code, result.timed_out) if not passed else result
