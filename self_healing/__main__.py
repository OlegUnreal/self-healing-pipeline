"""CLI entrypoint: `python -m self_healing --demo` / `--src … --test …`."""
from __future__ import annotations

import sys

from .cli import main


if __name__ == "__main__":
    sys.exit(main())
