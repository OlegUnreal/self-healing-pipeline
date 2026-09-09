"""Live demo: real LLM repairs a broken function.

Requires OPENAI_API_KEY. Falls back to stub if missing.
"""
from __future__ import annotations

import os
import sys
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
    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    proposer = make_openai_proposer() if has_key else stub_proposer
    mode = "OpenAI" if has_key else "stub (no OPENAI_API_KEY)"
    print(f"mode: {mode}")

    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "add.py"
        src.write_text(BROKEN)
        try:
            report = heal(src, TEST, proposer)
        except Exception as e:  # noqa: BLE001
            print(f"healer crashed: {e}", file=sys.stderr)
            sys.exit(1)

        for a in report.history:
            tag = "PASS" if a.passed else "fail"
            extra = f" err={a.error}" if a.error else ""
            print(f"attempt {a.n}: {tag}{extra}")
        print("success:", report.success)
        print("final source:\n", report.final_source)
        sys.exit(0 if report.success else 2)


if __name__ == "__main__":
    main()
