"""Verifier: re-runs the test and classifies the failure mode."""
from __future__ import annotations

from .sandbox import RunResult, run_code


def _classify(result: RunResult) -> str:
    """Return a coarse failure class for logging / LLM context."""
    if result.timed_out:
        return "timeout"
    err = (result.stderr or "") + (result.stdout or "")
    low = err.lower()
    if "syntaxerror" in low:
        return "syntax_error"
    if "indentationerror" in low:
        return "indentation_error"
    if "assertionerror" in low or "fail" in low:
        return "assertion_failure"
    if "importerror" in low or "modulenotfounderror" in low:
        return "import_error"
    if result.exit_code != 0:
        return "runtime_error"
    return "unknown"


def verify(test_code: str, timeout: float = 5.0) -> RunResult:
    """Run `test_code`; success == exit code 0 and no 'FAIL' anywhere.

    The returned RunResult.stderr is prefixed with a `[class=...]` tag so the
    LLM proposer sees *why* it failed, not just the raw traceback.
    """
    result = run_code(test_code, timeout=timeout)
    combined = (result.stdout or "") + (result.stderr or "")
    passed = result.exit_code == 0 and "FAIL" not in combined
    if passed:
        return result
    kind = _classify(result)
    tagged_err = f"[class={kind}]\n{result.stderr or result.stdout}"
    return RunResult(result.stdout, tagged_err, result.exit_code, result.timed_out, error=kind)
