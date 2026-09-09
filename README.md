# self-healing-pipeline

An agent that runs a failing test, reads the traceback, asks an LLM for a
unified diff, applies it safely, and repeats until green.

## Architecture

Four isolated layers:

- `sandbox.py` — executes untrusted code in a subprocess with resource limits
  (timeout, memory via `RLIMIT_AS`). Returns stdout, stderr, exit code.
- `patcher.py` — applies unified diffs, **rejects path traversal** (absolute
  paths, `..`), validates the result compiles with `py_compile`, rolls back on
  failure.
- `verifier.py` — re-runs the test suite and decides pass/fail.
- `agent.py` — the loop: observe failure → propose patch → apply → verify →
  repeat (max N). Empty diffs and LLM exceptions are skipped, not fatal.
- `llm.py` — real OpenAI hook with **retries, fence stripping, injectable
  client** for tests.

## Why this is interesting

This is the core loop behind Cursor / Claude Code, but built from scratch.
The hard parts are not the LLM calls — they are sandbox isolation, safe diff
application, and avoiding infinite repair loops. Every one of those is handled
here and covered by tests.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m self_healing.demo            # stub, no key needed
OPENAI_API_KEY=sk-... python -m self_healing.demo_llm   # real LLM
pytest -q
```

## Interview angle

On a whiteboard you can defend every decision: why a subprocess sandbox,
why `patch -p0` over string surgery, why retries belong in the proposer and
not the loop, why empty diffs must not burn the attempt budget.
