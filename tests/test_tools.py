from pathlib import Path

import pytest

from self_healing.agent import heal_with_tools
from self_healing.tools import Workspace, default_registry
from self_healing.tools.base import PathEscapeError


def test_workspace_rejects_escape(tmp_path):
    ws = Workspace(tmp_path)
    with pytest.raises(PathEscapeError):
        ws.resolve("../secret")
    with pytest.raises(PathEscapeError):
        ws.resolve("/etc/passwd")


def test_list_read_write_rollback(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    ws = Workspace(tmp_path)
    reg = default_registry()
    listed = reg.call(ws, "list_files", {"glob": "*.py"})
    assert listed.ok
    assert "a.py" in listed.data["files"]
    read = reg.call(ws, "read_file", {"path": "a.py"})
    assert "x = 1" in read.data["content"]
    written = reg.call(ws, "write_file", {"path": "a.py", "content": "x = 2\n"})
    assert written.ok
    assert (tmp_path / "a.py").read_text() == "x = 2\n"
    rolled = reg.call(ws, "rollback_file", {"path": "a.py"})
    assert rolled.ok
    assert (tmp_path / "a.py").read_text() == "x = 1\n"


def test_search_and_symbols(tmp_path):
    (tmp_path / "mod.py").write_text("def add(a, b):\n    return a + b\n\nclass Box:\n    def hold(self):\n        pass\n")
    ws = Workspace(tmp_path)
    reg = default_registry()
    hits = reg.call(ws, "search_text", {"pattern": "return a \\+ b", "path": "."})
    assert hits.ok and hits.data["hits"]
    symbols = reg.call(ws, "list_symbols", {"path": "mod.py"})
    names = {s["name"] for s in symbols.data["symbols"]}
    assert names == {"add", "Box"}
    extracted = reg.call(ws, "extract_function", {"path": "mod.py", "name": "add"})
    assert "return a + b" in extracted.data["source"]


def test_apply_patch_and_compile(tmp_path):
    src = tmp_path / "add.py"
    src.write_text("def add(a, b):\n    return a - b\n")
    ws = Workspace(tmp_path)
    reg = default_registry()
    diff = (
        "--- a/add.py\n+++ b/add.py\n@@ -1,2 +1,2 @@\n"
        " def add(a, b):\n-    return a - b\n+    return a + b\n"
    )
    patched = reg.call(ws, "apply_patch", {"path": "add.py", "diff": diff})
    assert patched.ok
    compiled = reg.call(ws, "compile_check", {"path": "add.py"})
    assert compiled.ok
    tests = reg.call(ws, "run_tests", {"test_code": "from add import add\nassert add(2,3)==5\n"})
    assert tests.ok and tests.data["passed"] is True


def test_unknown_tool(tmp_path):
    ws = Workspace(tmp_path)
    result = default_registry().call(ws, "launch_missiles", {})
    assert result.ok is False
    assert "unknown tool" in result.error


def test_openai_schemas_cover_all_tools():
    schemas = default_registry().schemas()
    names = {s["function"]["name"] for s in schemas}
    assert "apply_patch" in names
    assert "extract_function" in names
    assert "run_tests" in names
    assert len(names) >= 14


def test_heal_with_tools_scripted(tmp_path):
    src = tmp_path / "add.py"
    src.write_text("def add(a, b):\n    return a - b\n")
    test = "from add import add\nassert add(2, 3) == 5\n"
    diff = (
        "--- a/add.py\n+++ b/add.py\n@@ -1,2 +1,2 @@\n"
        " def add(a, b):\n-    return a - b\n+    return a + b\n"
    )
    seen: list[str] = []

    def planner(messages, schemas):
        if "apply_patch" not in seen:
            seen.append("apply_patch")
            return {"content": "fix", "tool_calls": [{"name": "apply_patch", "arguments": {"path": "add.py", "diff": diff}}]}
        return {"content": "done", "tool_calls": [{"name": "finish", "arguments": {}}]}

    report = heal_with_tools(src, test, planner, max_steps=5)
    assert report.success is True
    assert "return a + b" in report.final_source
    assert any(t["tool"] == "apply_patch" for t in report.tool_trace)
