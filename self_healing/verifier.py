"""Verifier: re-runs the test and classifies the failure mode."""
from __future__ import annotations

from pathlib import Path

from .sandbox import RunResult, run_code


def classify_output(result: RunResult) -> str:
    """Return a coarse failure class for logging / LLM context."""
    if result.timed_out:
        return "timeout"
    err = (result.stderr or "") + (result.stdout or "")
    low = err.lower()
    if "syntaxerror" in low:
        return "syntax_error"
    if "indentationerror" in low:
        return "indentation_error"
    if "importerror" in low or "modulenotfounderror" in low:
        return "import_error"
    if "nameerror" in low:
        return "name_error"
    if "assertionerror" in low or "\nfail" in low or low.startswith("fail"):
        return "assertion_failure"
    if "typeerror" in low:
        return "type_error"
    if result.exit_code != 0:
        return "runtime_error"
    return "ok"


def _classify(result: RunResult) -> str:
    return classify_output(result)


def verify(
    test_code: str,
    timeout: float = 5.0,
    cwd: str | Path | None = None,
    memory_mb: int = 256,
) -> RunResult:
    """Run `test_code`; success == exit code 0 and no 'FAIL' anywhere."""
    result = run_code(test_code, timeout=timeout, cwd=cwd, memory_mb=memory_mb)
    combined = (result.stdout or "") + (result.stderr or "")
    passed = result.exit_code == 0 and "FAIL" not in combined
    if passed:
        return result
    kind = _classify(result)
    tagged_err = f"[class={kind}]\n{result.stderr or result.stdout}"
    return RunResult(result.stdout, tagged_err, result.exit_code, result.timed_out, error=kind)
