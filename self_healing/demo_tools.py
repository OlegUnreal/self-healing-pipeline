"""Offline demo of the tool-calling repair loop (no API key)."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent import heal_with_tools

BROKEN = """\
def add(a, b):
    return a - b  # bug: should be +
"""

TEST = """\
from add import add
assert add(2, 3) == 5
print("ok")
"""

FIX = (
    "--- a/add.py\n+++ b/add.py\n@@ -1,2 +1,2 @@\n"
    " def add(a, b):\n-    return a - b  # bug: should be +\n+    return a + b\n"
)

PKG_UTIL_BROKEN = """\
def sanitize(text):
    return escape(text)
"""

PKG_UTIL_FIXED = """\
from html import escape


def sanitize(text):
    return escape(text)
"""

PKG_APP = """\
from util import sanitize


def banner(name):
    return sanitize(name)
"""

PKG_TEST = """\
from app import banner
assert banner("<x>") == "&lt;x&gt;"
print("ok")
"""

PKG_FIX = (
    "--- a/util.py\n+++ b/util.py\n@@ -1,2 +1,4 @@\n"
    "+from html import escape\n"
    "+\n"
    " def sanitize(text):\n"
    "     return escape(text)\n"
)


@dataclass(frozen=True)
class ScriptedFix:
    inspect_path: str
    patch_path: str
    diff: str
    test_code: str


ADD_FIX = ScriptedFix("add.py", "add.py", FIX, TEST)
PKG_SCRIPT = ScriptedFix("util.py", "util.py", PKG_FIX, PKG_TEST)


def make_scripted_planner(
    fix: ScriptedFix,
    test_code: str | None = None,
):
    """Deterministic planner used by demos and ``--planner stub``."""

    test_src = test_code if test_code is not None else fix.test_code

    def planner(messages: list[dict[str, Any]], schemas: list[dict[str, Any]]) -> dict[str, Any]:
        del schemas
        seen = [m.get("name") for m in messages if m.get("role") == "tool"]
        if "run_tests" not in seen:
            return {
                "content": "observe failing tests",
                "tool_calls": [{"name": "run_tests", "arguments": {"test_code": test_src}}],
            }
        if "read_file" not in seen:
            return {
                "content": "read source",
                "tool_calls": [{"name": "read_file", "arguments": {"path": fix.inspect_path}}],
            }
        if "list_symbols" not in seen:
            return {
                "content": "list symbols",
                "tool_calls": [{"name": "list_symbols", "arguments": {"path": fix.inspect_path}}],
            }
        if "apply_patch" not in seen:
            return {
                "content": "apply the known fix",
                "tool_calls": [{"name": "apply_patch", "arguments": {"path": fix.patch_path, "diff": fix.diff}}],
            }
        return {"content": "done", "tool_calls": [{"name": "finish", "arguments": {}}]}

    return planner


scripted_planner = make_scripted_planner(ADD_FIX)


def detect_scripted_fix(root: Path) -> ScriptedFix | None:
    """Pick a stub fixture from files already in the workspace jail."""
    add = root / "add.py"
    if add.is_file() and "return a - b" in add.read_text(encoding="utf-8", errors="replace"):
        return ADD_FIX
    util = root / "util.py"
    if util.is_file():
        text = util.read_text(encoding="utf-8", errors="replace")
        if "escape(" in text and "from html" not in text:
            return PKG_SCRIPT
    return None


def write_pkg_fixture(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "util.py").write_text(PKG_UTIL_BROKEN)
    (root / "app.py").write_text(PKG_APP)
    test = root / "test_app.py"
    test.write_text(PKG_TEST)
    return test


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "add.py"
        src.write_text(BROKEN)
        report = heal_with_tools(src, TEST, scripted_planner, max_steps=8)
        print("tools used:", [t["tool"] for t in report.tool_trace])
        for a in report:
            print(f"step {a.n}: passed={a.passed} tools={a.tools}")
        print("success:", report.success)
        print("final source:\n", report.final_source)


if __name__ == "__main__":
    main()
