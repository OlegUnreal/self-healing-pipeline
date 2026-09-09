"""Tests for the hardened agent loop."""
from __future__ import annotations

from pathlib import Path

from self_healing.agent import heal


def test_empty_diff_is_skipped(tmp_path):
    src = tmp_path / "add.py"
    src.write_text("def add(a, b):\n    return a - b\n")
    test = 'assert False, "FAIL"\n'
    calls = {"n": 0}

    def boom(_tb: str) -> str:
        calls["n"] += 1
        return "" if calls["n"] == 1 else (
            "--- a/add.py\n+++ b/add.py\n@@ -1,2 +1,2 @@\n"
            " def add(a, b):\n-    return a - b\n+    return a + b\n"
        )

    report = heal(src, test, boom, max_attempts=3)
    assert report.success is True
    assert report.attempts == 2
    assert "return a + b" in report.final_source


def test_llm_exception_surfaced(tmp_path):
    src = tmp_path / "x.py"
    src.write_text("x = 1\n")
    test = 'assert False, "FAIL"\n'

    def raiser(_tb: str) -> str:
        raise RuntimeError("api down")

    report = heal(src, test, raiser, max_attempts=2)
    assert report.success is False
    assert report.history[0].error == "api down"
