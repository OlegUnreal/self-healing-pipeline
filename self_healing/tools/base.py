"""Shared types for the tool layer.

Every tool runs inside a Workspace jail: paths that escape `root` are
rejected before any I/O. Results are JSON-serialisable so they can be
fed back to an LLM as tool-call output.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


class PathEscapeError(ValueError):
    """Raised when a tool argument tries to leave the workspace root."""


@dataclass
class Workspace:
    root: Path
    max_file_bytes: int = 200_000
    backups: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()

    def resolve(self, rel: str | Path) -> Path:
        raw = str(rel).strip() or "."
        if raw.startswith("/"):
            raise PathEscapeError(f"absolute paths are forbidden: {raw}")
        candidate = (self.root / raw).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as e:
            raise PathEscapeError(f"path escapes workspace: {raw}") from e
        return candidate

    def remember(self, path: Path) -> None:
        key = str(path.relative_to(self.root))
        if path.is_file() and key not in self.backups:
            self.backups[key] = path.read_text(encoding="utf-8", errors="replace")

    def restore(self, rel: str) -> str | None:
        path = self.resolve(rel)
        key = str(path.relative_to(self.root))
        if key not in self.backups:
            return None
        path.write_text(self.backups[key], encoding="utf-8")
        return self.backups[key]


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str = ""
    tool: str = ""

    def to_llm(self) -> str:
        payload = {"ok": self.ok, "tool": self.tool}
        if self.ok:
            payload["data"] = self.data
        else:
            payload["error"] = self.error
        return json.dumps(payload, default=str, ensure_ascii=False)[:12_000]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class Tool:
    spec: ToolSpec
    handler: Callable[..., ToolResult]

    def __call__(self, workspace: Workspace, **kwargs: Any) -> ToolResult:
        try:
            result = self.handler(workspace, **kwargs)
        except PathEscapeError as e:
            result = ToolResult(ok=False, error=str(e))
        except FileNotFoundError as e:
            result = ToolResult(ok=False, error=f"not found: {e}")
        except Exception as e:  # noqa: BLE001
            result = ToolResult(ok=False, error=f"{type(e).__name__}: {e}")
        result.tool = self.spec.name
        return result
