"""Resolve a CLI --src path into (primary file, jail root)."""
from __future__ import annotations

from pathlib import Path

PREFERRED_SOURCES = ("app.py", "main.py", "add.py", "util.py")


def resolve_workspace(src: Path) -> tuple[Path, Path]:
    """Return ``(primary_source, workspace_root)``.

    ``src`` may be a file (jail = its parent) or a directory (jail = that
    directory). Inside a directory we pick a primary module so reports still
    have a ``final_source`` path; the jail sees every file under the root.
    """
    target = Path(src).expanduser().resolve()
    if target.is_file():
        return target, target.parent
    if target.is_dir():
        for name in PREFERRED_SOURCES:
            candidate = target / name
            if candidate.is_file():
                return candidate, target
        pys = sorted(
            p for p in target.glob("*.py") if not p.name.startswith("test_")
        )
        if pys:
            return pys[0], target
        raise FileNotFoundError(f"no Python source in directory: {target}")
    raise FileNotFoundError(f"source not found: {target}")
