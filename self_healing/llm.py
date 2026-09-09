"""Real LLM hook for the repair loop.

Injectable: pass any callable with the same signature as `propose_diff`.
Default implementation calls OpenAI chat completions with retries and
robust fence stripping so a single flaky response never breaks the loop.
"""
from __future__ import annotations

import os
import re
import time
from typing import Callable


SYSTEM = (
    "You are a senior engineer fixing a failing Python test. "
    "Given the traceback, output ONLY a unified diff (---/+++) that fixes the bug. "
    "No explanation, no markdown fences, just the diff."
)

_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\n?|```$", re.MULTILINE)


def _strip_fences(text: str) -> str:
    """Remove accidental ``` fences the model may wrap around the diff."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.rstrip().endswith("```"):
        text = text.rstrip()[: -len("```")].rstrip()
    return text.strip()


def make_openai_proposer(
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
    client=None,
) -> Callable[[str], str]:
    """Return a propose_diff(traceback) -> str backed by OpenAI.

    `client` is injectable for tests; otherwise an OpenAI() is built from
    OPENAI_API_KEY. Retries on empty/transient responses with backoff.
    """
    if client is None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("pip install openai") from e
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def propose_diff(traceback: str) -> str:
        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.chat.completions.create(
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
            except Exception as e:  # noqa: BLE001 - retry transient API errors
                last_err = e
            time.sleep(0.5 * attempt)
        raise RuntimeError(f"LLM proposer failed after {max_retries} attempts: {last_err}")

    return propose_diff
