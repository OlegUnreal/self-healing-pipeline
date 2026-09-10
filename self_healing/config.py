"""Environment-backed settings. Secrets stay out of source and logs."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _load_dotenv() -> None:
    """Best-effort `.env` load. Missing python-dotenv is not fatal."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = ""
    model: str = "gpt-4o-mini"
    max_attempts: int = 5
    llm_timeout: float = 30.0
    sandbox_timeout: float = 5.0
    sandbox_memory_mb: int = 256
    max_tool_steps: int = 12
    max_file_bytes: int = 200_000


def load_settings() -> Settings:
    _load_dotenv()
    return Settings(
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        model=os.environ.get("SHP_MODEL", "gpt-4o-mini"),
        max_attempts=_int("SHP_MAX_ATTEMPTS", 5),
        llm_timeout=_float("SHP_TIMEOUT", 30.0),
        sandbox_timeout=_float("SHP_SANDBOX_TIMEOUT", 5.0),
        sandbox_memory_mb=_int("SHP_SANDBOX_MEMORY_MB", 256),
        max_tool_steps=_int("SHP_MAX_TOOL_STEPS", 12),
        max_file_bytes=_int("SHP_MAX_FILE_BYTES", 200_000),
    )
