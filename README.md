# self-healing-pipeline

An agent that runs a failing test, reads the traceback, asks an LLM for a unified diff, applies it safely, and repeats until green.

This is the core repair loop behind tools like Cursor and Claude Code — built from scratch in Python, with every hard part isolated and tested.

## Why this exists

Most "AI coding agent" demos are a single LLM call wrapped in a `while True`. The real difficulty is not the model — it is:

- **Sandbox isolation** — running untrusted code without poisoning the host.
- **Safe patch application** — applying a diff without path traversal or syntax corruption.
- **Failure classification** — telling the model *why* it failed, not just dumping a raw traceback.
- **Loop termination** — avoiding infinite repair loops and burning the attempt budget on empty diffs.

This repo handles all four, with tests for each.

## Architecture

Four isolated layers, each in its own module:

```
self_healing/
├── sandbox.py      # subprocess execution, RLIMIT_AS, timeout, host-error capture
├── patcher.py      # unified diff apply, path-traversal rejection, py_compile check, rollback
├── verifier.py     # re-run tests, classify failure (timeout/syntax/assertion/import/runtime)
├── agent.py        # observe → propose → apply → verify loop, EventLog on every failure
├── llm.py          # OpenAI hook: retries, fence stripping, injectable client for tests
├── logging.py      # structured JSON logs (no secrets, hashes instead of full source)
├── demo.py         # offline stub demo (no key)
└── demo_llm.py     # live demo with real OpenAI
```

### Layer responsibilities

| Module | Does | Does NOT |
|---|---|---|
| `sandbox.py` | Runs code in a subprocess with memory/time limits; returns `RunResult` | Never raises on host errors (missing interpreter, permission denied) |
| `patcher.py` | Applies unified diff via `patch -p0`, rejects `..` and absolute paths, validates with `py_compile`, rolls back on failure | Does not call the LLM |
| `verifier.py` | Re-runs the test suite, classifies the failure type, tags the traceback | Does not propose fixes |
| `agent.py` | Orchestrates the loop; skips empty diffs and LLM exceptions instead of crashing | Does not own sandbox/patcher/verifier internals |
| `llm.py` | Retries transient errors, strips markdown fences, accepts an injectable client | Does not know about diffs or tests |

## How to run

```bash
# 1. Clone
git clone https://github.com/OlegUnreal/self-healing-pipeline.git
cd self-healing-pipeline

# 2. Virtualenv
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4a. Offline demo (no API key needed)
python -m self_healing --demo

# 4b. Live demo (real OpenAI)
cp .env.example .env               # put OPENAI_API_KEY=sk-... in .env
python -m self_healing --demo-llm

# 5. Tests
pytest -q
```

Windows notes:

- Activate with `.venv\Scripts\activate`.
- `patch -p0` must be on PATH (Git for Windows ships it as `patch.exe`).
- The sandbox uses `RLIMIT_AS`, which is a no-op on Windows — memory caps are best-effort there.

## Libraries used and why

| Library | Version | Why it is here |
|---|---|---|
| `openai` | `>=1.40` | Official OpenAI Python SDK. Gives a typed client, streaming, retries, and tool/function calling without hand-rolled HTTP. The `llm.py` module depends only on this interface, so swapping to another provider is a one-file change. |
| `pytest` | `>=8.0` | The standard Python test runner. Used for unit tests of sandbox, patcher, verifier, and the agent loop. Fixtures make it trivial to inject a fake LLM client. |
| `python-dotenv` | `>=1.0` | Loads `.env` into the environment so secrets never live in source. Read once at startup in `config.py`. |

No other runtime dependencies on purpose: the sandbox shells out to the system `patch` and `python` binaries, and the agent is plain stdlib orchestration. Fewer dependencies means fewer supply-chain surprises and a smaller attack surface.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | — | required for `--demo-llm` |
| `SHP_MODEL` | `gpt-4o-mini` | model used by the proposer |
| `SHP_MAX_ATTEMPTS` | `5` | max repair iterations before giving up |
| `SHP_TIMEOUT` | `30` | per-LLM-call timeout (seconds) |
| `SHP_SANDBOX_MEMORY_MB` | `256` | `RLIMIT_AS` cap for the sandbox subprocess |

All settings are read from the environment or a `.env` file (via `python-dotenv`). Secrets are never written to logs.

## Testing

```bash
pytest -q                 # all tests, quiet
pytest -v tests/test_patcher.py   # one module
pytest --cov=self_healing # coverage (optional)
```

Tests cover: sandbox host-error capture, path-traversal rejection, `py_compile` rollback, failure classification, empty-diff skipping, LLM retry behaviour, and log scrubbing.

## Design decisions (interview notes)

These are the questions an interviewer will ask, and the answers baked into the code:

1. **Why a subprocess sandbox instead of `exec()`?**
   `exec()` runs in the same process — a malicious or buggy patch can `import os; os.remove(...)` or monkey-patch your own modules. A subprocess with `RLIMIT_AS` isolates memory and lets the OS kill runaway processes.

2. **Why `patch -p0` over string surgery?**
   String replacement breaks on whitespace and context drift. `patch` is the same tool humans use in code review; it fails loudly on mismatch, which the agent then reports back to the LLM.

3. **Why classify failures instead of dumping raw tracebacks?**
   A raw traceback is noisy and the model wastes tokens re-deriving the cause. Classification (`timeout` / `syntax` / `assertion` / `import` / `runtime`) lets the prompt say "this is a syntax error, fix the indentation" — far more actionable.

4. **Why retries in the proposer, not the loop?**
   Transient API errors (429, 5xx) should be retried at the call site. Putting retries in the loop would burn attempt budget on network blips. The loop only counts *logical* failures.

5. **Why skip empty diffs instead of counting them as attempts?**
   An empty diff means the model declined to change anything. Counting it as an attempt punishes the agent for a non-event and exhausts the budget. Skipping it preserves the budget for real proposals.

## Project status

Working prototype: real LLM integration, structured logging, full test suite, offline stub mode. Not a production product — no persistent job queue, no multi-file repo support, no web UI. Suitable as a portfolio piece and as a base to extend.

## License

MIT.
