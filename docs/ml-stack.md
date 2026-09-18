# The ML stack

Two independent learning layers sit on top of the repair loop. Both are
optional (`pip install -e ".[ml]"`), both fail soft, and neither one is
allowed to change a repair verdict — they only remember and suggest.

```
run (red) ──► verifier ──► [class=...] ──► planner prompt
                 │                              ▲
                 ▼                              │ recall (top-k similar past repairs)
           RepairMemory ◄── observe(diff, verdict) ──┘
                 ▲
        FailureClassifier ──► failure_class tag + confidence gate
```

## Layer 1 — repair memory (a small vector store)

`self_healing/memory.py`. A SQLite table of `(traceback, diff, success,
failure_class, meta)` rows plus a chosen embedder, brute-scored at recall
time. No server, no index file — the whole store is one `.sqlite3` file
pointed at by `SHP_MEMORY_PATH`.

- Embedders: `hashing` (default; blake2-bucketed signed hashing, no training)
  or `tfidf-lsa` (fitted, used by the classifier).
- Recall: cosine over stored rows, optionally filtered by failure class,
  re-ranked with a prior that favours rows whose repair was green in fewer
  attempts.
- Seeding: on first open an empty store is filled with `REPAIR_EXAMPLES`
  — a corpus of curated (traceback, diff) pairs — so `--ml` is useful
  from the first run.

### What is remembered, and when

Two granularities, one per loop type:

| Loop | Unit of memory | Tag |
|---|---|---|
| `diff` (proposer loop) | one row per attempt: the patch it produced and the verdict the *next* verify returns for it | `loop="diff"` |
| `tools` / `graph` | one row per episode: `Workspace.net_diff()` — the unified diff rebuilt from pre-run backups, i.e. the net effect of all tool calls | `loop="tools"` / `loop="graph"` |

The diff loop settles rows lazily: attempt *n* writes a `pending` tuple,
and the verdict for it is delivered by the verify at the top of attempt
*n+1* (plus one extra verify after the loop for the final patch). A patch
that goes green on the last attempt is still learnt — the post-loop verify
runs regardless of the budget.

A tool-loop repair reaches its result through ten separate calls, so the
unit worth remembering is the net effect on the workspace. Files created
mid-run have no backup and are absent from `net_diff()`; the edits that
fix real code are the ones that carry information.

### Memory observes, it does not steer

Every learning hook is gated on `memory is not None`. No recall outcome,
no classifier label, and no memory error can flip `report.success` or a
verdict. The invariant is pinned by `tests/test_memory_wiring.py`:
`test_verdicts_repeat_but_memory_never_changes_them` (identical verdict
sequences with and without a store) and
`test_without_memory_the_graph_never_pays_for_net_diff` (a workspace
without memory never even builds a diff).

### The PYTHONHASHSEED lesson

An earlier draft of the hashing embedder bucketed features with Python's
built-in `hash()`. That silently poisons a persistent store: `hash()` is
salted per process (the `PYTHONHASHSEED` randomisation), so vectors
written by run A were unfindable by run B — recall returned nothing and
the failure mode was invisible. The fix, and the rule:

> anything persisted next to vectors must be keyed by a stable hash
> (`hashlib.blake2b`), never by `hash()`. Determinism must come from the
> algorithm, not from an env var.

`HashingEmbedder` now buckets via blake2b and carries a `version` field,
so a future bucket-scheme change can invalidate old rows explicitly.

## Layer 2 — the failure classifier

`self_healing/classifier.py`. Logistic regression over `tfidf-lsa`
embeddings of the run transcript, with an order-aware "structured view"
that keeps the *sequence* of exception names. Trained on the shipped
corpus (`SEED_TRACEBACKS + HARD_TRACEBACKS`, 189 rows, 8 classes).

### Training and evaluation

```bash
pip install -e ".[ml]"
self-healing --train-classifier                 # artifact -> SHP_ML_MODEL_PATH
self-healing --train-classifier --model-path my-artifact.joblib
```

Deterministic: stratified split seed 7, SVD seed 7. Same input, same
artifact (within floating-point noise).

Full metrics are in [`model-card.md`](model-card.md). Headlines:

| Metric | Model | Keyword rules |
|---|---|---|
| macro-F1 (56-row held-out test) | **0.8379** | 0.6975 |
| chained tracebacks (n=35) | **0.7315** | 0.5022 |
| single-exception traces (n=21) | 1.0 | 1.0 |
| expected calibration error | 0.2055 | — |

Ablation (`bag_only_macro_f1`): switching off every order-sensitive
feature (word bigrams + char 3–5 grams, unigrams only) drops macro-F1
from 0.8379 to 0.7302. The order features — which exception precedes
which — are worth more than any single vocabulary item. That is also why
the model beats the rules exactly where the rules are weakest: chained
tracebacks, where the *last* exception is not the interesting one.

### The OOV / ablation lesson

Two traps found while building the evaluation:

1. **The control must be a real bag of words.** The first ablation
   reused the production vectorizer on raw text, which kept word bigrams
   and char n-grams — both encode exception order — so the "control"
   secretly measured the same thing. `_bag_ablation()` now fits a
   bespoke unigram-only `tfidf-lsa` (see the docstring at
   `classifier.py:244`).
2. **Word-level OOV is real; char-level OOV is not.** `TfidfVectorizer`
   drops unseen vocabulary at transform time, so a traceback from a
   library never seen in training loses its word bigrams. The
   `char_wb` 3–5 gram half of the union carries those rows instead,
   which is why the embedder is a word+char concatenation rather than
   either half alone.

### The confidence gate: defer to the rules

`verifier.classify_with_ml(result, classifier, *, min_confidence=0.35)`
answers `None` — handing the decision back to the keyword ladder — when:

- the transcript is empty;
- the classifier raises (wrong sklearn, unfitted artifact, missing numpy);
- it says `ok` while the exit code is non-zero;
- the top probability is below `min_confidence`.

A missing or broken model must leave the rules in charge, never take the
repair loop down with the exception. Timeouts are returned as `timeout`
before the model is consulted.

## Using it end to end

```bash
self-healing --train-classifier                 # once; caches the artifact
self-healing --src examples/add.py \
             --test "from add import add; assert add(2,3)==5" \
             --mode tools --ml --yes            # recall + learned classes on
```

What you will see: a `memory.recalled` log line on the first red run,
hints appended to the planner prompt as a user turn (exactly once per
episode), and a `memory: {...}` stats block at the end of the report.

| Variable | Default | Meaning |
|---|---|---|
| `SHP_USE_ML` / `--ml` | off | enable memory + classifier |
| `SHP_ML_MODEL_PATH` / `--model-path` | `models/failure_classifier.joblib` | artifact path |
| `SHP_MEMORY_PATH` | `.shp/repair-memory.sqlite3` | store file |
| `SHP_MEMORY_TOP_K` | `3` | hints injected per episode |

Without `numpy`/`scikit-learn` installed, `open_memory()` /
`open_classifier()` return `None`, the loops run unchanged, and the CLI
prints a note — the ML stack degrades to the 0.3.x behaviour.
