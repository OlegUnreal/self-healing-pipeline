"""Real LLM hook for the repair loop.

Injectable: pass any callable with the same signature as `propose_diff`.
Default implementation calls OpenAI chat completions.
"""
from __future__ import annotations

import os
from typing import Callable


SYSTEM = (
    "You are a senior engineer fixing a failing Python test. "
    "Given the traceback, output ONLY a unified diff (---/+++) that fixes the bug. "
    "No explanation, no markdown fences, just the diff."
)


def make_openai_proposer(model: str = "gpt-4o-mini") -> Callable[[str], str]:
    """Return a propose_diff(traceback) -> str backed by OpenAI."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("pip install openai") from e

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def propose_diff(traceback: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": traceback[:6000]},
            ],
            temperature=0,
        )
        text = resp.choices[0].message.content or ""
        # Strip accidental markdown fences the model may add.
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
        return text.strip()

    return propose_diff
