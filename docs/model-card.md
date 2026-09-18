# Failure classifier model card

- backend: `logistic` over `tfidf-lsa` embeddings (dim 96)
- classes: 8
- train/test: 133/56 stratified split
- accuracy **0.8393**, macro-F1 **0.8379** (rule baseline 0.6975)
- expected calibration error: 0.2055
- chained tracebacks (n=35): model 0.7315 vs rules 0.5022
- single-exception traces (n=21): model 1.0 vs rules 1.0
- ablation: exception-order features move macro-F1 from 0.7302 (bag of words) to 0.8379

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| `assertion_failure` | 0.8571 | 0.8571 | 0.8571 | 7 |
| `import_error` | 0.875 | 0.875 | 0.875 | 8 |
| `indentation_error` | 0.875 | 1.0 | 0.9333 | 7 |
| `name_error` | 1.0 | 0.8571 | 0.9231 | 7 |
| `runtime_error` | 0.6667 | 0.8571 | 0.75 | 7 |
| `syntax_error` | 1.0 | 0.8571 | 0.9231 | 7 |
| `timeout` | 0.8 | 0.6667 | 0.7273 | 6 |
| `type_error` | 0.7143 | 0.7143 | 0.7143 | 7 |
