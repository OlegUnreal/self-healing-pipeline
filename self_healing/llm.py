"""Real LLM hook for the repair loop — diffs and tool-calling."""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable

SYSTEM = (
    "You are a senior engineer fixing a failing Python test. "
    "Given the traceback, output ONLY a unified diff (---/+++) that fixes the bug. "
    "No explanation, no markdown fences, just the diff."
)

TOOL_SYSTEM = (
    "You are a senior engineer repairing a failing Python workspace. "
    "Use the provided tools to inspect files, search symbols, apply a unified "
    "diff, compile, and re-run tests. Prefer apply_patch over rewriting whole files. "
    "When tests pass, stop calling tools."
)

_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\n?|```$", re.MULTILINE)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.rstrip().endswith("```"):
        text = text.rstrip()[: -len("```")].rstrip()
    return text.strip()


class _CompatClient:
    def __init__(self, client: Any):
        self._raw = client

    def create(self, **kwargs: Any) -> Any:
        raw = self._raw
        if hasattr(raw, "chat") and hasattr(raw.chat, "completions"):
            return raw.chat.completions.create(**kwargs)
        if callable(raw) and not hasattr(raw, "chat"):
            msgs = kwargs.get("messages", [])
            content = msgs[-1]["content"] if msgs else ""
            out = raw(content)
            text = out if isinstance(out, str) else str(out)
            return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": text, "tool_calls": None})()})()]})()
        if hasattr(raw, "chat") and callable(raw.chat):
            chat = raw.chat()
            comp = chat.completions() if hasattr(chat, "completions") else chat
            return comp.create(**kwargs)
        if hasattr(raw, "create") and callable(raw.create):
            return raw.create(**kwargs)
        raise TypeError(f"unsupported client type: {type(raw)!r}")

    @property
    def chat(self) -> "_CompatClient":
        return self

    @property
    def completions(self) -> "_CompatClient":
        return self


def _build_client(client: Any | None) -> _CompatClient:
    if client is None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("pip install openai") from e
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _CompatClient(client)


def make_openai_proposer(model: str = "gpt-4o-mini", max_retries: int = 3, client=None) -> Callable[[str], str]:
    compat = _build_client(client)

    def propose_diff(traceback: str) -> str:
        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = compat.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": traceback[:6000]},
                    ],
                    temperature=0,
                )
                text = _strip_fences(resp.choices[0].message.content or "")
                if text:
                    return text
                last_err = ValueError("empty model response")
            except Exception as e:  # noqa: BLE001
                last_err = e
            time.sleep(0.5 * attempt)
        raise RuntimeError(f"LLM proposer failed after {max_retries} attempts: {last_err}")

    return propose_diff


def make_openai_planner(model: str = "gpt-4o-mini", max_retries: int = 3, client=None):
    compat = _build_client(client)

    def planner(messages: list[dict[str, Any]], schemas: list[dict[str, Any]]) -> dict[str, Any]:
        last_err: Exception | None = None
        payload = [{"role": "system", "content": TOOL_SYSTEM}, *messages]
        for attempt in range(1, max_retries + 1):
            try:
                kwargs: dict[str, Any] = {"model": model, "messages": payload, "temperature": 0}
                if schemas:
                    kwargs["tools"] = schemas
                    kwargs["tool_choice"] = "auto"
                resp = compat.create(**kwargs)
                msg = resp.choices[0].message
                calls = []
                raw_calls = getattr(msg, "tool_calls", None) or []
                for raw in raw_calls:
                    fn = getattr(raw, "function", raw)
                    name = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else "")
                    args = getattr(fn, "arguments", None) or (fn.get("arguments") if isinstance(fn, dict) else {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args else {}
                        except json.JSONDecodeError:
                            args = {}
                    calls.append({"name": name, "arguments": args})
                return {"content": getattr(msg, "content", "") or "", "tool_calls": calls}
            except Exception as e:  # noqa: BLE001
                last_err = e
            time.sleep(0.5 * attempt)
        raise RuntimeError(f"LLM planner failed after {max_retries} attempts: {last_err}")

    return planner
