"""Tool registry: schemas for the LLM + dispatch for the runtime."""
from __future__ import annotations

from typing import Any

from . import handlers
from .base import Tool, ToolResult, ToolSpec, Workspace


def _obj(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        schema["required"] = required
    return schema


def _str(desc: str) -> dict[str, Any]:
    return {"type": "string", "description": desc}


def _int(desc: str) -> dict[str, Any]:
    return {"type": "integer", "description": desc}


def _num(desc: str) -> dict[str, Any]:
    return {"type": "number", "description": desc}


_TOOLS: list[Tool] = [
    Tool(ToolSpec("workspace_info", "Describe the current workspace jail (root, size limits, backups).", _obj({})), handlers.workspace_info),
    Tool(ToolSpec("list_files", "List files under a workspace-relative path. Use glob like '*.py' or '**/*'.", _obj({"path": _str("Directory relative to workspace."), "glob": _str("Filename glob.")})), handlers.list_files),
    Tool(ToolSpec("read_file", "Read a text file with line numbers. Optionally slice start_line/end_line.", _obj({"path": _str("File path relative to workspace."), "start_line": _int("1-based start line."), "end_line": _int("Inclusive end line.")}, ["path"])), handlers.read_file),
    Tool(ToolSpec("write_file", "Overwrite or create a text file inside the workspace. Previous content is backed up.", _obj({"path": _str("File path relative to workspace."), "content": _str("Full file contents.")}, ["path", "content"])), handlers.write_file),
    Tool(ToolSpec("file_stat", "Return size, type and mtime for a workspace path.", _obj({"path": _str("Path relative to workspace.")}, ["path"])), handlers.file_stat),
    Tool(ToolSpec("apply_patch", "Apply a unified diff to one file. Rejects path traversal and rolls back on syntax errors.", _obj({"path": _str("Target file."), "diff": _str("Unified diff text.")}, ["path", "diff"])), handlers.apply_patch),
    Tool(ToolSpec("rollback_file", "Restore a file from the in-memory backup taken before write_file/apply_patch.", _obj({"path": _str("File to restore.")}, ["path"])), handlers.rollback_file),
    Tool(ToolSpec("compile_check", "Run py_compile on a Python file without executing it.", _obj({"path": _str("Python file.")}, ["path"])), handlers.compile_check),
    Tool(ToolSpec("run_python", "Execute a short Python snippet in the sandbox (cwd = workspace root).", _obj({"code": _str("Python source to run."), "timeout": _num("Seconds.")}, ["code"])), handlers.run_python),
    Tool(ToolSpec("run_tests", "Run the given test snippet in the sandbox and classify the failure.", _obj({"test_code": _str("Python test source."), "timeout": _num("Seconds.")}, ["test_code"])), handlers.run_tests),
    Tool(ToolSpec("search_text", "Regex search across workspace files. Defaults to *.py.", _obj({"pattern": _str("Python regular expression."), "path": _str("Start directory."), "glob": _str("Filename glob."), "max_hits": _int("Cap on returned hits.")}, ["pattern"])), handlers.search_text),
    Tool(ToolSpec("list_symbols", "Parse a Python file with ast and list top-level functions and classes.", _obj({"path": _str("Python file.")}, ["path"])), handlers.list_symbols),
    Tool(ToolSpec("extract_function", "Surgically extract one function by name using ast.get_source_segment.", _obj({"path": _str("Python file."), "name": _str("Function name.")}, ["path", "name"])), handlers.extract_function),
    Tool(ToolSpec("classify_failure", "Map a traceback or test output to a coarse class (syntax, assertion, import, ...).", _obj({"output": _str("Traceback or test output.")}, ["output"])), handlers.classify_failure),
    Tool(ToolSpec("git_status", "Return git status --porcelain if the workspace is a git repo.", _obj({})), handlers.git_status),
    Tool(ToolSpec("git_diff", "Return git diff for the workspace or a single path.", _obj({"path": _str("Optional path filter.")})), handlers.git_diff),
]


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools = {t.spec.name: t for t in (tools or _TOOLS)}

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.spec.openai_schema() for t in self._tools.values()]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def call(self, workspace: Workspace, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, error=f"unknown tool: {name}", tool=name)
        args = dict(arguments or {})
        return tool(workspace, **args)


def default_registry() -> ToolRegistry:
    return ToolRegistry()
