# Python tools

Every tool is a `(Workspace, **args) -> ToolResult` function exposed to the
LLM as an OpenAI function schema. Paths are workspace-relative. Absolute
paths and `..` are rejected before I/O.

Unknown tool names return `ok=False` instead of raising.

The graph orchestrator (`heal_with_graph`) calls this same registry. It does
not grow a second set of tools. See `docs/langgraph.md`.

| Tool | Why it exists | How it is wired |
|---|---|---|
| `workspace_info` | Jail root and backups. | `Workspace` metadata. |
| `list_files` | Orientation. | `Path.rglob` with skip-dirs. |
| `read_file` | Inspect with line numbers. | Size cap `max_file_bytes`. |
| `write_file` | Last-resort rewrite. | Backup first. |
| `file_stat` | Existence / size. | `Path.stat`. |
| `apply_patch` | Preferred mutation. | `patcher.apply_diff`. |
| `rollback_file` | Undo last write/patch. | `Workspace.backups`. |
| `compile_check` | Catch SyntaxError early. | `python -m py_compile`. |
| `run_python` | Sandboxed snippet. | `sandbox.run_code`. |
| `run_tests` | Ground truth for the loop. | `verifier.verify`. |
| `search_text` | Find call sites. | Regex + glob. |
| `list_symbols` | Map a file. | `ast.parse`. |
| `extract_function` | Send only the failing function. | `ast.get_source_segment`. |
| `classify_failure` | Coarse class for routing. | Same classifier as `verifier`. |
| `git_status` / `git_diff` | Optional VCS context. | `git`. |
