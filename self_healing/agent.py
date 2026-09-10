"""The repair loop: observe → propose → apply → verify, up to max_attempts.

A second loop, `heal_with_tools`, lets the model call the sandboxed Python
tool registry instead of emitting a raw diff in one shot.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from . import logging as logmod
from .patcher import apply_diff
from .tools import ToolRegistry, Workspace, default_registry
from .verifier import verify

log = logmod.get_logger(__name__)
