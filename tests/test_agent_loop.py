"""Tests for the self-healing repair loop."""
from __future__ import annotations

from pathlib import Path

from self_healing.agent import heal
from self_healing.verifier import verify


def test_heal_succeeds_with_good_diff(tmp_path):
    src = tmp_path / "add.py"
    src.write_text("def add(a, b):\n    return a - b\n")
    test = 'from add import add\nassert add(2, 3) == 5, "FAIL"\n'
    diff = (
        "--- a/add.py\n+++ b/add.py\n@@ -1,2 +1,2 @@\n"
        " def add(a, b):\n-    return a - b\n+    return a + b\n"
    )
    report = heal(src, test, lambda _tb: diff, max_attempts=3)
    assert report.success is True
    assert "return a + b" in report.final_source
    assert report.attempts >= 1


def test_heal_skips_empty_diff(tmp_path):
    src = tmp_path / "x.py"
    src.write_text("x = 1\n")
    test = 'assert False, "FAIL"\n'
    calls = {"n": 0}

    def flaky(_tb: str) -> str:
        calls["n"] += 1
        return "" if calls["n"] == 1 else (
            "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"
        )

    report = heal(src, test, flaky, max_attempts=3)
    assert report.success is True
    assert report.attempts == 2


def test_heal_survives_proposer_crash(tmp_path):
    src = tmp_path / "x.py"
    src.write_text("x = 1\n")
    test = 'assert False, "FAIL"\n'
    report = heal(src, test, lambda _tb: (_ for _ in ()).throw(RuntimeError("api down")), max_attempts=2)
    assert report.success is False
    assert "proposer failed" in report.history[0].error


def test_heal_survives_verify_crash(tmp_path, monkeypatch):
    src = tmp_path / "x.py"
    src.write_text("x = 1\n")

    def boom_verify(_test: str, timeout: float = 5.0):
        raise RuntimeError("verifier down")

    monkeypatch.setattr("self_healing.agent.verify", boom_verify)
    report = heal(src, "assert False", lambda _tb: "", max_attempts=1)
    assert report.success is False
    assert "verifier crashed" in report.history[0].error


def test_verify_tags_failure_class():
    r = verify('raise AssertionError("x")')
    assert r.exit_code != 0
    assert r.stderr.startswith("[class=assertion_failure]")


def test_verify_passes_on_green():
    r = verify('assert 1 + 1 == 2')
    assert r.exit_code == 0
    assert r.timed_out is False
