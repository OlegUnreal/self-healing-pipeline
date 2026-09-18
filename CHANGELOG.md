# Changelog

## 0.5.0 — 2026-09-18

- `--ml`: repair memory and the failure classifier wired into every loop (diff, tools, graph) and the CLI; `SHP_USE_ML=1` is equivalent.
- Learning granularities: the diff loop stores one row per attempt (lazy verdict settle — the patch from attempt *n* is judged by the verify at attempt *n+1*, plus one post-loop verify so a last-attempt green patch is still learnt); tool/graph loops store one episode row via `Workspace.net_diff()`, tagged `loop="diff"|"tools"|"graph"` in row meta.
- First red run recalls top-k similar past repairs into the planner prompt, exactly once per episode (`memory.recalled` log line).
- New tools `recall_repairs` / `classify_failure`; `--train-classifier` / `--model-path`; memory stats block at the end of every report.
- `classify_with_ml` confidence gate (`min_confidence=0.35`): empty transcript, model error, `ok`-vs-nonzero-exit and low confidence all defer to the keyword ladder. Timeouts are decided before the model is consulted.
- 25 wiring tests pin the invariant: memory observes, it does not steer — identical verdict sequences with and without a store; a memory-less workspace never even builds a net diff.
- Docs: [`ml-stack.md`](docs/ml-stack.md) (training/eval, the PYTHONHASHSEED and bag-of-words-ablation lessons, the two learning granularities) and [`model-card.md`](docs/model-card.md).

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
