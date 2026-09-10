from pathlib import Path

import pytest

from self_healing.tools import Workspace, default_registry
from self_healing.tools.base import PathEscapeError


def test_rejects_parent_and_absolute(tmp_path):
    ws = Workspace(tmp_path)
    with pytest.raises(PathEscapeError):
        ws.resolve("../secret")
    with pytest.raises(PathEscapeError):
        ws.resolve("/etc/passwd")
    with pytest.raises(PathEscapeError):
        ws.resolve("foo/../../etc/passwd")


def test_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("classified\n")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not supported")
    ws = Workspace(tmp_path)
    with pytest.raises(PathEscapeError):
        ws.resolve("link.txt")
    result = default_registry().call(ws, "read_file", {"path": "link.txt"})
    assert result.ok is False
    assert "escape" in result.error or "forbidden" in result.error


def test_read_rejects_oversized_file(tmp_path):
    ws = Workspace(tmp_path, max_file_bytes=32)
    (tmp_path / "big.py").write_text("x" * 64)
    result = default_registry().call(ws, "read_file", {"path": "big.py"})
    assert result.ok is False
    assert "too large" in result.error


def test_write_rejects_oversized_content(tmp_path):
    ws = Workspace(tmp_path, max_file_bytes=16)
    result = default_registry().call(ws, "write_file", {"path": "a.py", "content": "y" * 64})
    assert result.ok is False
    assert "max_file_bytes" in result.error


def test_apply_patch_invalid_diff_keeps_file(tmp_path):
    src = tmp_path / "add.py"
    src.write_text("def add(a, b):\n    return a - b\n")
    ws = Workspace(tmp_path)
    result = default_registry().call(
        ws, "apply_patch", {"path": "add.py", "diff": "this is not a unified diff"}
    )
    assert result.ok is False
    assert src.read_text() == "def add(a, b):\n    return a - b\n"


def test_apply_patch_traversal_in_header_rejected(tmp_path):
    src = tmp_path / "add.py"
    src.write_text("def add(a, b):\n    return a - b\n")
    ws = Workspace(tmp_path)
    diff = "--- a/../../etc/passwd\n+++ b/../../etc/passwd\n@@ -1 +1 @@\n-x\n+y\n"
    result = default_registry().call(ws, "apply_patch", {"path": "add.py", "diff": diff})
    assert result.ok is False
    assert src.read_text().startswith("def add")


def test_unknown_tool_does_not_raise(tmp_path):
    result = default_registry().call(Workspace(tmp_path), "rm_rf", {})
    assert result.ok is False
    assert "unknown tool" in result.error
