# Changelog

## 0.3.0 — 2026-09-10

- Tool-calling loop with a sandboxed Python tool registry.
- Graph orchestrator (`heal_with_graph`): Observe → Plan → Tools → Observe.
- Optional LangGraph runtime (`pip install -r requirements-langgraph.txt`).
- Workspace CLI: `--src` / `--test` / `--mode diff|tools|graph`.
- Mutation gate: `--approve-mutations` (+ `--yes`).
- Hardened jail tests: path escape, symlink, size caps, bad diffs.

## 0.2.0

- Diff loop, sandbox, patcher, verifier, OpenAI proposer/planner.
