"""Unit tests for the hardened LLM layer (no network)."""
from __future__ import annotations

import self_healing.llm as llm


class FakeResp:
    def __init__(self, content: str):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})]


class FakeClient:
    """Mimics openai.OpenAI: client.chat.completions.create(...)."""

    def __init__(self, contents):
        self._contents = list(contents)
        self.calls = 0

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **_):
        self.calls += 1
        return FakeResp(self._contents.pop(0))


def test_strip_fences():
    assert llm._strip_fences("```diff\n--- a/x\n+++ b/x\n```") == "--- a/x\n+++ b/x"


def test_retries_then_returns():
    client = FakeClient(["", "--- a/x\n+++ b/x\n"])
    proposer = llm.make_openai_proposer(client=client)
    assert proposer("tb") == "--- a/x\n+++ b/x\n"
    assert client.calls == 2


def test_raises_after_exhaustion():
    client = FakeClient(["", "", ""])
    proposer = llm.make_openai_proposer(client=client, max_retries=2)
    try:
        proposer("tb")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "failed after 2 attempts" in str(e)
