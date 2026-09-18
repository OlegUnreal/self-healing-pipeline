"""Concrete tool implementations.

These are the only side-effecting operations the model is allowed to
request. Each handler receives a Workspace so path jail + backups stay
centralised.
"""
from __future__ import annotations

import ast
import fnmatch
import re
import subprocess
import sys
from typing import Any

from ..patcher import apply_diff
from ..sandbox import run_code
from ..verifier import classify_output, verify
from .base import ToolResult, Workspace

_SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules"}


def list_files(ws: Workspace, path: str = ".", glob: str = "**/*") -> ToolResult:
    root = ws.resolve(path)
    if not root.exists():
        return ToolResult(ok=False, error=f"missing path: {path}")
    if root.is_file():
        return ToolResult(ok=True, data={"files": [path]})
    files: list[str] = []
    for item in sorted(root.rglob("*")):
        if any(part in _SKIP_DIRS for part in item.parts):
            continue
        if not item.is_file():
            continue
        rel = str(item.relative_to(ws.root))
        if glob and glob not in ("*", "**/*") and not fnmatch.fnmatch(rel, glob):
            if not fnmatch.fnmatch(item.name, glob):
                continue
        files.append(rel)
        if len(files) >= 500:
            break
    return ToolResult(ok=True, data={"files": files, "count": len(files)})


def read_file(
    ws: Workspace,
    path: str,
    start_line: int = 1,
    end_line: int | None = None,
) -> ToolResult:
    target = ws.resolve(path)
    if not target.is_file():
        return ToolResult(ok=False, error=f"not a file: {path}")
    if target.stat().st_size > ws.max_file_bytes:
        return ToolResult(ok=False, error=f"file too large (>{ws.max_file_bytes} bytes)")
    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    start = max(1, int(start_line))
    end = int(end_line) if end_line is not None else len(lines)
    end = min(end, len(lines))
    numbered = [f"{i:>4}|{lines[i - 1]}" for i in range(start, end + 1)]
    return ToolResult(
        ok=True,
        data={
            "path": path,
            "start_line": start,
            "end_line": end,
            "total_lines": len(lines),
            "content": "\n".join(numbered),
        },
    )


def write_file(ws: Workspace, path: str, content: str) -> ToolResult:
    target = ws.resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if len(content.encode("utf-8")) > ws.max_file_bytes:
        return ToolResult(ok=False, error="content exceeds max_file_bytes")
    if target.exists():
        ws.remember(target)
    target.write_text(content, encoding="utf-8")
    return ToolResult(ok=True, data={"path": path, "bytes": len(content.encode("utf-8"))})


def file_stat(ws: Workspace, path: str) -> ToolResult:
    target = ws.resolve(path)
    if not target.exists():
        return ToolResult(ok=False, error=f"missing path: {path}")
    st = target.stat()
    return ToolResult(
        ok=True,
        data={
            "path": path,
            "is_file": target.is_file(),
            "is_dir": target.is_dir(),
            "size": st.st_size,
            "mtime": int(st.st_mtime),
        },
    )


def apply_patch(ws: Workspace, path: str, diff: str) -> ToolResult:
    target = ws.resolve(path)
    if not target.is_file():
        return ToolResult(ok=False, error=f"not a file: {path}")
    ws.remember(target)
    ok = apply_diff(target, diff)
    if not ok:
        return ToolResult(ok=False, error="patch rejected (mismatch, traversal, or syntax)")
    return ToolResult(ok=True, data={"path": path, "applied": True})


def rollback_file(ws: Workspace, path: str) -> ToolResult:
    restored = ws.restore(path)
    if restored is None:
        return ToolResult(ok=False, error=f"no backup for {path}")
    return ToolResult(ok=True, data={"path": path, "restored": True, "bytes": len(restored)})


def compile_check(ws: Workspace, path: str) -> ToolResult:
    target = ws.resolve(path)
    if not target.is_file():
        return ToolResult(ok=False, error=f"not a file: {path}")
    proc = subprocess.run(
        [sys.executable, "-m", "py_compile", str(target)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return ToolResult(ok=False, error=proc.stderr.strip() or proc.stdout.strip())
    return ToolResult(ok=True, data={"path": path, "compiles": True})


def run_python(ws: Workspace, code: str, timeout: float = 5.0) -> ToolResult:
    result = run_code(code, timeout=timeout, cwd=ws.root)
    return ToolResult(
        ok=result.exit_code == 0 and not result.timed_out,
        data={
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
            "class": classify_output(result),
        },
        error="" if result.exit_code == 0 else (result.error or result.stderr[:500]),
    )


def run_tests(ws: Workspace, test_code: str, timeout: float = 5.0) -> ToolResult:
    result = verify(test_code, timeout=timeout, cwd=ws.root)
    passed = result.exit_code == 0 and not result.timed_out
    return ToolResult(
        ok=passed,
        data={
            "passed": passed,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
            "class": result.error or classify_output(result),
        },
        error="" if passed else (result.stderr or result.stdout)[:1500],
    )


def search_text(
    ws: Workspace,
    pattern: str,
    path: str = ".",
    glob: str = "*.py",
    max_hits: int = 40,
) -> ToolResult:
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return ToolResult(ok=False, error=f"invalid regex: {e}")
    root = ws.resolve(path)
    hits: list[dict[str, Any]] = []
    files = [root] if root.is_file() else sorted(root.rglob("*"))
    for item in files:
        if not item.is_file():
            continue
        if any(part in _SKIP_DIRS for part in item.parts):
            continue
        if glob and not fnmatch.fnmatch(item.name, glob) and not fnmatch.fnmatch(str(item), glob):
            continue
        try:
            text = item.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                hits.append({"path": str(item.relative_to(ws.root)), "line": i, "text": line[:240]})
                if len(hits) >= max_hits:
                    return ToolResult(ok=True, data={"hits": hits, "truncated": True})
    return ToolResult(ok=True, data={"hits": hits, "truncated": False})


def list_symbols(ws: Workspace, path: str) -> ToolResult:
    target = ws.resolve(path)
    if not target.is_file():
        return ToolResult(ok=False, error=f"not a file: {path}")
    source = target.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return ToolResult(ok=False, error=f"syntax error: {e}")
    symbols: list[dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(
                {
                    "kind": "function",
                    "name": node.name,
                    "lineno": node.lineno,
                    "end_lineno": getattr(node, "end_lineno", node.lineno),
                    "args": [a.arg for a in node.args.args],
                }
            )
        elif isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            symbols.append(
                {
                    "kind": "class",
                    "name": node.name,
                    "lineno": node.lineno,
                    "end_lineno": getattr(node, "end_lineno", node.lineno),
                    "methods": methods,
                }
            )
    return ToolResult(ok=True, data={"path": path, "symbols": symbols})


def extract_function(ws: Workspace, path: str, name: str) -> ToolResult:
    target = ws.resolve(path)
    if not target.is_file():
        return ToolResult(ok=False, error=f"not a file: {path}")
    source = target.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return ToolResult(ok=False, error=f"syntax error: {e}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            snippet = ast.get_source_segment(source, node) or ""
            return ToolResult(
                ok=True,
                data={
                    "path": path,
                    "name": name,
                    "lineno": node.lineno,
                    "end_lineno": getattr(node, "end_lineno", node.lineno),
                    "source": snippet,
                },
            )
    return ToolResult(ok=False, error=f"function not found: {name}")


def classify_failure(ws: Workspace, output: str) -> ToolResult:
    from ..sandbox import RunResult

    dummy = RunResult(stdout=output, stderr="", exit_code=1, timed_out="timeout" in output.lower())
    classifier = getattr(ws, "classifier", None)
    if classifier is not None:
        from ..verifier import classify_with_ml

        learned = classify_with_ml(dummy, classifier)
        if learned:
            return ToolResult(ok=True, data={"class": learned, "backend": "learned", "workspace": str(ws.root)})
    return ToolResult(ok=True, data={"class": classify_output(dummy), "backend": "rules", "workspace": str(ws.root)})


def recall_repairs(ws: Workspace, query: str, k: int = 3) -> ToolResult:
    """Ask the episodic store what fixed a similar traceback before.

    The scores are exposed rather than just the diffs: a planner that can see
    `similarity` next to `prior` is in a better position to distrust a near-random
    hit than one handed three unlabelled patches.
    """
    memory = getattr(ws, "memory", None)
    if memory is None:
        return ToolResult(ok=False, error="repair memory disabled (SHP_USE_ML=1, SHP_MEMORY_PATH=...)")
    try:
        hits = memory.recall(query, max(1, int(k)))
    except Exception as e:  # noqa: BLE001 - a broken index is not a repair blocker
        return ToolResult(ok=False, error=f"recall failed: {type(e).__name__}: {e}")
    return ToolResult(
        ok=True,
        data={
            "count": len(hits),
            "repairs": [
                {
                    "class": h.record.failure_class,
                    "similarity": round(h.similarity, 4),
                    "prior": round(h.prior, 4),
                    "score": round(h.score, 4),
                    "green": h.record.success,
                    "attempts": h.record.attempts,
                    "diff": h.record.diff[:4000],
                }
                for h in hits
            ],
        },
    )


def git_status(ws: Workspace) -> ToolResult:
    git_dir = ws.root / ".git"
    if not git_dir.exists():
        return ToolResult(ok=False, error="workspace is not a git repository")
    proc = subprocess.run(
        ["git", "status", "--porcelain", "-uall"],
        cwd=ws.root,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if proc.returncode != 0:
        return ToolResult(ok=False, error=proc.stderr.strip() or "git status failed")
    return ToolResult(ok=True, data={"porcelain": proc.stdout.splitlines()})


def git_diff(ws: Workspace, path: str = "") -> ToolResult:
    git_dir = ws.root / ".git"
    if not git_dir.exists():
        return ToolResult(ok=False, error="workspace is not a git repository")
    cmd = ["git", "diff", "--", path] if path else ["git", "diff"]
    proc = subprocess.run(cmd, cwd=ws.root, capture_output=True, text=True, timeout=5)
    if proc.returncode != 0:
        return ToolResult(ok=False, error=proc.stderr.strip() or "git diff failed")
    return ToolResult(ok=True, data={"diff": proc.stdout[:8000]})


def workspace_info(ws: Workspace) -> ToolResult:
    return ToolResult(
        ok=True,
        data={
            "root": str(ws.root),
            "max_file_bytes": ws.max_file_bytes,
            "backups": list(ws.backups.keys()),
        },
    )
