# LangGraph extra

The core package does not depend on LangGraph. `heal_with_graph()` always
runs through the same nodes:

```
START → observe → plan → tools → observe → …
                 ↘ escalate → END
                 → END when tests are green
```

When `langgraph` is installed *and* you pass `use_langgraph=True` (or
`--use-langgraph`), those nodes are compiled into a real `StateGraph`.
Otherwise a local interpreter walks the same functions so CI and
`--demo-graph` stay dependency-free.

## Why LangGraph at all

| Concern | Owner |
|---|---|
| path jail, sandbox, patch, classify | this repo |
| routing, budget, stream, resume, HITL | graph layer |

LangGraph is useful when a repair job must survive a process restart
(checkpointer), pause before `apply_patch` (`--approve-mutations`), or
stream node updates to a UI.

It is the wrong default for the unit tests: they must not need an API
key or a graph library.

## Install

```bash
pip install -r requirements.txt
pip install -r requirements-langgraph.txt
# or
pip install -e ".[langgraph]"
```

## Run

```bash
python -m self_healing --demo-graph
python -m self_healing --src examples/add.py --test examples/test_add.py \
    --mode graph --planner stub --stream
python -m self_healing --src examples/add.py --test examples/test_add.py \
    --mode graph --use-langgraph --checkpoint /tmp/heal.sqlite --thread-id job-1
```

`--approve-mutations` blocks `write_file` / `apply_patch` until the
approver returns true. In a TTY that is a prompt; pass `--yes` to
auto-approve. Without a TTY, mutations are rejected unless `--yes`.

## Mapping

| Graph node | Existing code |
|---|---|
| `observe` | `verifier.verify` |
| `plan` | `Planner` (`make_openai_planner` or `scripted_planner`) |
| `tools` | `ToolRegistry.call` inside `Workspace` |
| `escalate` | `HealReport.error` |

`Workspace` and `ToolRegistry` are **not** stored in graph state. They
live on `GraphContext`. State only holds JSON-serialisable fields.

`create_react_agent` is intentionally unused. A prebuilt ReAct loop
would hide classify / budget / jail, which is the point of this repo.
