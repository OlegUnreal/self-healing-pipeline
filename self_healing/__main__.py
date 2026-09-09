"""CLI entrypoint: `python -m self_healing --demo` or `--demo-llm`."""
from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="self_healing")
    parser.add_argument("--demo", action="store_true", help="run stub demo (no key)")
    parser.add_argument("--demo-llm", action="store_true", help="run real LLM demo")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.verbose:
        import logging
        logging.getLogger("self_healing").setLevel(logging.DEBUG)

    if args.demo_llm:
        from .demo_llm import main as run
    elif args.demo:
        from .demo import main as run
    else:
        parser.print_help()
        return 2

    try:
        run()
    except BrokenPipeError:
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"fatal: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
