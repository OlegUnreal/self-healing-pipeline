# self-healing-pipeline

An agent that writes a failing test, runs it, reads the traceback, patches the source, and repeats until green.

## Architecture

Three isolated layers:

- `sandbox/` — executes untrusted code in a subprocess with resource limits (timeout, memory). Returns stdout, stderr, exit code.
- `patcher/` — applies unified diffs to files, validates the result compiles, rolls back on failure.
- `verifier/` — re-runs the test suite and decides pass/fail.
- `agent/` — the loop: observe failure → propose patch → apply → verify → repeat (max N attempts).

## Why this is interesting

This is the core loop behind Cursor / Claude Code, but built from scratch. The hard parts are not the LLM calls — they are sandbox isolation, safe diff application, and avoiding infinite repair loops.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m self_healing.demo
```
