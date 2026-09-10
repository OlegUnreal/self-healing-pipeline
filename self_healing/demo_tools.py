"""Offline demo of the tool-calling repair loop (no API key)."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from .agent import heal_with_tools


BROKEN = '''\
def add(a, b):
    return a - b  # bug: should be +
'''

TEST = '''\
from add import add
assert add(2, 3) == 5
print("ok")
'''

FIX = (
    "--- a/add.py\n+++ b/add.py\n@@ -1,2 +1,2 @@\n"
    " def add(a, b):\n-    return a - b  # bug: should be +\n+    return a + b\n"
)


def scripted_planner(messages: list[dict[str, Any]], schemas: list[dict[str, Any]]) -> dict[str, Any]:
    tool_names_seen = [m.get("name") for m in messages if m.get("role") == "tool"]
    if "run_tests" not in tool_names_seen:
        return {"content": "observe failing tests", "tool_calls": [{"name": "run_tests", "arguments": {"test_code": TEST}}]}
    if "read_file" not in tool_names_seen:
        return {"content": "read source", "tool_calls": [{"name": "read_file", "arguments": {"path": "add.py"}}]}
    if "list_symbols" not in tool_names_seen:
        return {"content": "list symbols", "tool_calls": [{"name": "list_symbols", "arguments": {"path": "add.py"}}]}
    if "apply_patch" not in tool_names_seen:
        return {"content": "apply the arithmetic fix", "tool_calls": [{"name": "apply_patch", "arguments": {"path": "add.py", "diff": FIX}}]}
    return {"content": "done", "tool_calls": [{"name": "finish", "arguments": {}}]}


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
