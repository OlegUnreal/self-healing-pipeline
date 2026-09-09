"""Diff applier: applies a unified diff and validates it compiles.

Hardened against path traversal: absolute paths, `..` segments and paths
escaping the target's parent directory are rejected before `patch` runs.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

_PATH_RE = re.compile(r"^[ab]/(.+)")


def _validate_paths(diff: str, root: Path) -> bool:
    """Return False if any diff path is absolute or escapes `root`."""
    for line in diff.splitlines():
        if not (line.startswith("--- ") or line.startswith("+++ ")):
            continue
        raw = line[4:].split("\t", 1)[0].strip()
        if raw in ("/dev/null", ""):
            continue
        m = _PATH_RE.match(raw)
        rel = m.group(1) if m else raw.lstrip("/")
        if rel.startswith("/") or rel.startswith("..") or "/../" in f"/{rel}/":
            return False
        try:
            (root / rel).resolve().relative_to(root.resolve())
        except ValueError:
            return False
    return True


def _normalise_diff(diff: str, target: Path) -> str:
    """Rewrite `a/<name>` / `b/<name>` headers to the target's real filename.

    Models (and our own tests) emit diffs against `a/add.py` even when the
    file lives at `/tmp/.../add.py`. `patch -p0` matches on the literal path,
    so we strip the `a/`/`b/` prefix down to the basename.
    """
    out: list[str] = []
    for line in diff.splitlines():
        if line.startswith("--- ") or line.startswith("+++ "):
            prefix = line[:4]
            rest = line[4:]
            name = rest.split("\t", 1)[0].strip()
            m = _PATH_RE.match(name)
            rel = m.group(1) if m else name.lstrip("/")
            line = f"{prefix}{Path(rel).name}"
        out.append(line)
    return "\n".join(out) + "\n"


def apply_diff(target: Path, diff: str) -> bool:
    """Apply `diff` to `target`. Returns True on success.

    Strategy: validate paths, normalise headers to the target basename, write
    the diff to a temp file, run `patch -p0 --directory <parent>` so patch
    resolves the basename inside the target's own directory, then
    syntax-check with `py_compile`. Roll back on any failure.
    """
    if not diff or not diff.strip():
        return False
    if not _validate_paths(diff, target.parent):
        return False

    diff = _normalise_diff(diff, target)
    backup = target.read_text(encoding="utf-8")
    with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False) as f:
        f.write(diff)
        diff_path = f.name
    try:
        proc = subprocess.run(
            [
                "patch",
                "-p0",
                "--forward",
                "--batch",
                "--directory",
                str(target.parent),
                "--input",
                diff_path,
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            target.write_text(backup, encoding="utf-8")
            return False
        check = subprocess.run(
            [sys.executable, "-m", "py_compile", str(target)],
            capture_output=True,
            text=True,
        )
        if check.returncode != 0:
            target.write_text(backup, encoding="utf-8")
            return False
        return True
    finally:
        Path(diff_path).unlink(missing_ok=True)
