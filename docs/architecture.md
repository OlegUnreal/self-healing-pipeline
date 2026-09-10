# Architecture

Self-healing-pipeline is a closed control loop around a failing Python workspace.
The model never writes to disk itself. It only *proposes* actions. A jail
(`Workspace`) plus a sandbox (`run_code`) execute those actions.

## Control loop

```mermaid
flowchart TD
    A[Broken workspace + failing test] --> B[Observe]
    B --> C{Tests green?}
    C -->|yes| Z[Done: HealReport.success]
    C -->|no| D[Classify failure]
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
```

## Layers

```mermaid
flowchart LR
    subgraph Orchestration
      agent[agent.py]
      config[config.py]
      logs[logging.py]
    end
    subgraph Tools
      registry[ToolRegistry]
      fs[list/read/write/stat]
      ast[list_symbols / extract_function]
      search[search_text]
      exec[run_python / run_tests]
      patch[apply_patch / rollback]
      git[git_status / git_diff]
    end
    subgraph Isolation
      ws[Workspace jail]
      sand[sandbox + RLIMIT_AS]
      patcher[patcher + py_compile]
    end
    subgraph Model
      llm[llm.py OpenAI]
    end
    agent --> registry
    agent --> llm
    registry --> ws
    ws --> fs
    ws --> ast
    ws --> search
    ws --> exec
    ws --> patch
    ws --> git
    exec --> sand
    patch --> patcher
    config --> agent
    logs --> agent
```

## Sequence of a tool-healed repair

```mermaid
sequenceDiagram
    participant T as Test suite
    participant A as Agent
    participant M as Planner / LLM
    participant R as ToolRegistry
    participant S as Sandbox

    T->>A: failing traceback
    A->>M: messages + tool schemas
    M->>A: run_tests
    A->>R: run_tests(test_code)
    R->>S: python -c in workspace cwd
    S-->>A: class=assertion_failure
    M->>A: read_file + list_symbols
    A->>R: inspect add.py
    M->>A: apply_patch(diff)
    A->>R: patch -p0 + py_compile
    M->>A: finish
    A->>S: re-run tests
    S-->>A: exit 0
    A-->>T: HealReport.success
```

The model is an untrusted proposer. Tools are the capability boundary.
