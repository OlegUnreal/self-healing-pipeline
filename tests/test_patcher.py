"""Tests for the hardened patcher."""
from __future__ import annotations

from pathlib import Path

from self_healing.patcher import _validate_paths, apply_diff


def test_rejects_absolute_path():
    diff = "--- a//etc/passwd\n+++ b//etc/passwd\n@@ -1 +1 @@\n-x\n+y\n"
    assert _validate_paths(diff, Path("/tmp")) is False


def test_rejects_dotdot():
    diff = "--- a/../../etc/passwd\n+++ b/../../etc/passwd\n@@ -1 +1 @@\n-x\n+y\n"
    assert _validate_paths(diff, Path("/tmp/proj")) is False


def test_accepts_relative():
    diff = "--- a/add.py\n+++ b/add.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n"
    assert _validate_paths(diff, Path("/tmp/proj")) is True


def test_apply_roundtrip(tmp_path):
    src = tmp_path / "add.py"
    src.write_text("def add(a, b):\n    return a - b\n")
    diff = (
        "--- a/add.py\n+++ b/add.py\n@@ -1,2 +1,2 @@\n"
        " def add(a, b):\n-    return a - b\n+    return a + b\n"
    )
    assert apply_diff(src, diff) is True
    assert "return a + b" in src.read_text()
