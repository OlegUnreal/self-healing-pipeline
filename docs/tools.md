# Python tools

Every tool is a `(Workspace, **args) -> ToolResult` function exposed to the
LLM as an OpenAI function schema. Paths are workspace-relative. Absolute
paths and `..` are rejected before I/O.

| Tool | Why it exists | How it is wired |
|---|---|---|
| `workspace_info` | The model needs to know the jail root and which files already have backups. | Reads `Workspace` metadata only. |
| `list_files` | Orientation. Without a file list the model guesses names. | `Path.rglob` with skip-dirs. |
| `read_file` | Inspect the failing source with line numbers so diffs line up. | UTF-8 read, size cap `max_file_bytes`. |
| `write_file` | Last-resort full rewrite when a diff cannot be formed. | Backs up first, then writes. |
| `file_stat` | Cheap existence / size check before a read. | `Path.stat`. |
| `apply_patch` | Preferred mutation: same format humans review. | Delegates to `patcher.apply_diff`. |
| `rollback_file` | Undo a bad write/patch without restarting the workspace. | In-memory `Workspace.backups`. |
| `compile_check` | Catch `SyntaxError` before spending a test run. | `python -m py_compile`. |
| `run_python` | Reproduce a snippet, print values, sanity-check a hypothesis. | `sandbox.run_code` with `cwd=workspace`. |
| `run_tests` | Ground truth for the loop. | `verifier.verify` + failure class. |
| `search_text` | Find call sites of the broken symbol across files. | Compiled regex + glob. |
| `list_symbols` | Map a file without dumping the whole source. | `ast.parse` of top-level defs. |
| `extract_function` | Send only the failing function to the model (smaller, safer). | `ast.get_source_segment`. |
| `classify_failure` | Turn a raw traceback into a coarse class for routing. | Same classifier as `verifier`. |
| `git_status` | See what already changed if the workspace is a git repo. | `git status --porcelain`. |
| `git_diff` | Review the current working tree before another patch. | `git diff`. |

Unknown tool names return `ok=False` instead of raising, so a hallucinated
function name cannot crash the loop.
