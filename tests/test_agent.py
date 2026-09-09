"""Tests for the hardened agent loop."""
from __future__ import annotations

from self_healing.agent import heal


def test_empty_diff_is_skipped(tmp_path):
    src = tmp_path / "add.py"
    src.write_text("def add(a, b):\n    return a - b\n")
    # The flaky proposer's second answer flips `-` to `+`, so the assertion
    # must actually hold afterwards — `assert False` can never be healed.
    test = 'assert add(2, 3) == 5\n'
    calls = {"n": 0}

    def flaky(_tb: str) -> str:
        calls["n"] += 1
        return "" if calls["n"] == 1 else (
            "--- a/add.py\n+++ b/add.py\n@@ -1,2 +1,2 @@\n"
            " def add(a, b):\n-    return a - b\n+    return a + b\n"
        )

    report = heal(src, test, flaky, max_attempts=3)
    assert report.success is True
    assert report.attempts == 2
    assert "return a + b" in report.final_source
    assert report.events.events, "expected structured events"


def test_llm_exception_surfaced(tmp_path):
    src = tmp_path / "x.py"
    src.write_text("x = 1\n")
    test = 'assert False, "FAIL"\n'

    def raiser(_tb: str) -> str:
        raise RuntimeError("api down")

    report = heal(src, test, raiser, max_attempts=2)
    assert report.success is False
    assert report.history[0].error == "proposer failed: api down"
    assert "propose_crash" in {e["event"] for e in report.events.events}


def test_apply_exception_surfaced(tmp_path, monkeypatch):
    src = tmp_path / "x.py"
    src.write_text("x = 1\n")
    test = 'assert False, "FAIL"\n'

    def bad_patcher(_tb: str) -> str:
        raise ValueError("boom in patcher")

    monkeypatch.setattr("self_healing.agent.apply_diff", bad_patcher)
    report = heal(src, test, lambda _tb: "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n", max_attempts=1)
    assert report.success is False
    assert "patcher crashed" in report.history[0].error


def test_verify_crash_surfaced(tmp_path, monkeypatch):
    src = tmp_path / "x.py"
    src.write_text("x = 1\n")

    def boom_verify(_test: str, timeout: float = 5.0):
        raise RuntimeError("verifier down")

    monkeypatch.setattr("self_healing.agent.verify", boom_verify)
    report = heal(src, "assert False", lambda _tb: "", max_attempts=1)
    assert report.success is False
    assert "verifier crashed" in report.history[0].error
