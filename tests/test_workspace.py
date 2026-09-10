from pathlib import Path

import pytest

from self_healing.workspace import resolve_workspace


def test_resolve_file_uses_parent(tmp_path):
    src = tmp_path / "add.py"
    src.write_text("x = 1\n")
    primary, root = resolve_workspace(src)
    assert primary == src.resolve()
    assert root == tmp_path.resolve()


def test_resolve_directory_prefers_app(tmp_path):
    (tmp_path / "util.py").write_text("def sanitize(text): return text\n")
    (tmp_path / "app.py").write_text("from util import sanitize\n")
    (tmp_path / "test_app.py").write_text("assert True\n")
    primary, root = resolve_workspace(tmp_path)
    assert primary.name == "app.py"
    assert root == tmp_path.resolve()


def test_resolve_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_workspace(tmp_path / "nope.py")
