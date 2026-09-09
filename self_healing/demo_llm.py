"""Live demo: real LLM repairs a broken function.

Requires OPENAI_API_KEY. Falls back to stub if missing.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .agent import heal
from .llm import make_openai_proposer


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
    return (
        "--- a/add.py\n+++ b/add.py\n@@ -1,2 +1,2 @@\n"
        " def add(a, b):\n-    return a - b  # bug: should be +\n+    return a + b\n"
    )


def main() -> None:
    proposer = make_openai_proposer() if os.environ.get("OPENAI_API_KEY") else stub_proposer
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "add.py"
        src.write_text(BROKEN)
        history = heal(src, TEST, proposer)
        for a in history:
            print(f"attempt {a.n}: passed={a.passed}")
        print("final source:\n", src.read_text())


if __name__ == "__main__":
    main()
