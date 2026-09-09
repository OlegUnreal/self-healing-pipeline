"""Diff applier: applies a unified diff and validates it compiles."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def apply_diff(target: Path, diff: str) -> bool:
    """Apply `diff` to `target`. Returns True on success.

    Strategy: write the diff to a temp file, run `patch -p0`, then syntax-check
    the result with `py_compile`. Roll back on any failure.
    """
    backup = target.read_text(encoding="utf-8")
    with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False) as f:
        f.write(diff)
        diff_path = f.name
    try:
        proc = subprocess.run(
            ["patch", "-p0", "--forward", "--input", diff_path],
            cwd=target.parent,
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
