# Changelog

## 0.3.1 — 2026-09-10

- `--src` accepts a file or a directory (jail root). Primary module is picked from `app.py` / `main.py` / `add.py` / `util.py`.
- Second fixture `examples/pkg/` (`NameError`: missing `from html import escape`) — not arithmetic.
- Stub planner is fixture-detected, not hard-coded to `add.py`.
- `--checkpoint` with `--use-langgraph` raises `CheckpointerUnavailable` instead of silently dropping the saver.
- LangGraph HITL: mutating tools call `interrupt()` when the extra runtime is active.
- Classifier checks `NameError` / `ImportError` before the loose `fail` substring.

## 0.3.0 — 2026-09-10

- Tool-calling loop with a sandboxed Python tool registry.
- Graph orchestrator (`heal_with_graph`): Observe → Plan → Tools → Observe.
- Optional LangGraph runtime (`pip install -r requirements-langgraph.txt`).
- Workspace CLI: `--src` / `--test` / `--mode diff|tools|graph`.
- Mutation gate: `--approve-mutations` (+ `--yes`).
- Hardened jail tests: path escape, symlink, size caps, bad diffs.

## 0.2.0

- Diff loop, sandbox, patcher, verifier, OpenAI proposer/planner.
