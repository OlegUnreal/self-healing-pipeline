# Architecture

```mermaid
graph TD
    A[Broken code + failing test] --> B[Sandbox]
    B --> C[Verifier]
    C -->|FAIL| D[LLM propose_diff]
    D --> E[Patcher]
    E --> B
    C -->|PASS| F[Fixed code]
```

Three isolated layers:

| Layer | Responsibility |
|---|---|
| **Sandbox** | Runs untrusted code in a subprocess with memory limits |
| **Verifier** | Classifies failures: timeout, syntax, assertion |
| **Patcher** | Applies unified diffs, rejects path traversal |

The LLM never touches the filesystem directly — it only proposes diffs.
