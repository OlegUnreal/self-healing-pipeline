"""Structured logging for the repair loop.

JSON lines to stdout, no secrets, no full source dumps — only hashes and
lengths. Safe to ship to production log aggregators.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.time(),
            "level": record.levelname,
            "event": record.getMessage(),
            "logger": record.name,
        }
        for k, v in record.__dict__.items():
            if k in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "color_message",
            }:
                continue
            payload[k] = v
        return json.dumps(payload, default=str, ensure_ascii=False)


def get_logger(name: str = "self_healing") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]


@contextmanager
def attempt_span(logger: logging.Logger, n: int, **extra: Any) -> Iterator[None]:
    t0 = time.time()
    logger.info("attempt.start", attempt=n, **extra)
    try:
        yield
    finally:
        logger.info("attempt.end", attempt=n, elapsed_ms=round((time.time() - t0) * 1000), **extra)


class EventLog:
    """Tiny structured event buffer attached to a HealReport for debugging."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def add(self, event: str, **fields: Any) -> None:
        self.events.append({"event": event, "ts": time.time(), **fields})

    def as_json(self) -> str:
        return json.dumps(self.events, default=str, ensure_ascii=False)
