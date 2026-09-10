# Demo transcript

Recorded from `python -m self_healing --demo-graph` (no API key).

```
langgraph extra: no (local interpreter)
  node=observe   passed=False class=assertion_failure steps=0
  node=plan      passed=False class=assertion_failure steps=1
  node=tools     passed=False class=assertion_failure steps=1
  node=observe   passed=False class=assertion_failure steps=1
  node=plan      passed=False class=assertion_failure steps=2
  node=tools     passed=False class=assertion_failure steps=2
  node=observe   passed=False class=assertion_failure steps=2
  node=plan      passed=False class=assertion_failure steps=3
  node=tools     passed=False class=assertion_failure steps=3
  node=observe   passed=False class=assertion_failure steps=3
  node=plan      passed=False class=assertion_failure steps=4
  node=tools     passed=False class=assertion_failure steps=4
  node=observe   passed=True  class=ok steps=4
tools used: ['run_tests', 'read_file', 'list_symbols', 'apply_patch']
success: True
final source:
 def add(a, b):
    return a + b
```

Same repair via the workspace CLI:

```
python -m self_healing \
  --src examples/add.py \
  --test examples/test_add.py \
  --mode graph \
  --planner stub \
  --stream
```
