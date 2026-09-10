# Architecture

Self-healing-pipeline is a closed control loop around a failing Python workspace.
The model never writes to disk itself. It only *proposes* actions. A jail
(`Workspace`) plus a sandbox (`run_code`) execute those actions.

There are three orchestrators on top of the same capability boundary:

1. **Diff loop** (`heal`) — one unified diff per attempt.
2. **Tool loop** (`heal_with_tools`) — OpenAI-style function calls.
3. **Graph loop** (`heal_with_graph`) — Observe → Plan → Tools, with an optional LangGraph runtime.

## Control loop

```mermaid
flowchart TD
    A[Broken workspace + failing test] --> B[Observe]
    B --> C{Tests green?}
    C -->|yes| Z[Done: HealReport.success]
    C -->|no + budget left| D[Classify failure]
    C -->|budget exhausted| X[Escalate]
    D --> E{Mode}
    E -->|diff loop| F[LLM propose_diff]
    F --> G[Patcher]
    G --> H[py_compile + rollback]
    H --> B
    E -->|tool loop| I[LLM planner + tool schemas]
    I --> J[ToolRegistry.dispatch]
    J --> K[Workspace jail]
    K --> L[read / search / ast / patch / run]
    L --> B
    E -->|graph loop| M[plan node]
    M --> N[tools node]
    N --> B
    X --> Z2[HealReport.error]
```

## Graph nodes

```mermaid
flowchart TD
    START --> Observe
    Observe -->|passed| END
    Observe -->|budget gone| Escalate
    Observe -->|else| Plan
    Plan -->|tool_calls| Tools
    Plan -->|no calls| Observe
    Plan -->|planner crash| Escalate
    Tools --> Observe
    Escalate --> END
```

See also `docs/langgraph.md`.

The model is an untrusted proposer. Tools are the capability boundary.
The graph is only the router.
