"""Verifier: re-runs the test and classifies the failure mode."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .sandbox import RunResult, run_code

log = logging.getLogger("self_healing.verifier")


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


def classify_with_ml(result: RunResult, classifier: Any, *, min_confidence: float = 0.35) -> str | None:
    """Ask the trained model, and answer None when it has no standing to override.

    Returning None is the important half: the ML stack is optional, so a missing
    numpy, an unfitted artifact or an unsure top class must leave the keyword
    ladder in charge rather than take the repair loop down with the exception.
    """
    if result.timed_out:
        return "timeout"
    text = (result.stderr or "") + (result.stdout or "")
    if not text.strip():
        return None
    try:
        label = str(classifier.predict(text))
        top = float(classifier.predict_proba([text])[0].max())
    except Exception as exc:  # optional dependency absent or artifact unusable
        log.warning("ml classification unavailable, using rules", error=str(exc))
        return None
    if label == "ok" and result.exit_code != 0:
        return None
    if top < min_confidence:
        log.info("ml classification withheld", label=label, confidence=round(top, 4))
        return None
    return label


def _classify(result: RunResult, classifier: Any = None) -> str:
    if classifier is not None:
        learned = classify_with_ml(result, classifier)
        if learned:
            return learned
    return classify_output(result)


def verify(
    test_code: str,
    timeout: float = 5.0,
    cwd: str | Path | None = None,
    memory_mb: int = 256,
    classifier: Any = None,
) -> RunResult:
    """Run `test_code`; success == exit code 0 and no 'FAIL' anywhere."""
    result = run_code(test_code, timeout=timeout, cwd=cwd, memory_mb=memory_mb)
    combined = (result.stdout or "") + (result.stderr or "")
    passed = result.exit_code == 0 and "FAIL" not in combined
    if passed:
        return result
    kind = _classify(result, classifier)
    tagged_err = f"[class={kind}]\n{result.stderr or result.stdout}"
    return RunResult(result.stdout, tagged_err, result.exit_code, result.timed_out, error=kind)
