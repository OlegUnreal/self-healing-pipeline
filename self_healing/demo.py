"""End-to-end demo with a stub proposer (no LLM needed)."""
from __future__ import annotations

import tempfile
from pathlib import Path

from .agent import heal


BROKEN = '''\
def add(a, b):
    return a - b  # bug: should be +
'''

TEST = '''\
from add import add
assert add(2, 3) == 5, "FAIL: 2+3"
print("ok")
'''


def stub_proposer(traceback: str) -> str:
    # In production this calls an LLM. Here we hardcode the obvious fix.
    return (
        "--- a/add.py\n+++ b/add.py\n@@ -1,2 +1,2 @@\n"
        " def add(a, b):\n-    return a - b  # bug: should be +\n+    return a + b\n"
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "add.py"
        src.write_text(BROKEN)
        history = heal(src, TEST, stub_proposer)
        for a in history:
            print(f"attempt {a.n}: passed={a.passed}")
        print("final source:", src.read_text())


if __name__ == "__main__":
    main()
