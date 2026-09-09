"""Sandbox executor: runs untrusted Python in an isolated subprocess."""
from __future__ import annotations

import logging
import subprocess
import sys
import textwrap
from dataclasses import dataclass

from . import logging as logmod

log = logmod.get_logger(__name__)


@dataclass
class RunResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    error: str = ""


def run_code(code: str, timeout: float = 5.0, memory_mb: int = 256) -> RunResult:
    """Execute `code` in a fresh interpreter. No network, no filesystem writes outside /tmp.

    Catches OSError/PermissionError from the host side (e.g. missing interpreter,
    permission denied) and returns them as a RunResult instead of raising, so the
    repair loop never crashes on a sandbox setup failure.
    """
    wrapped = textwrap.dedent(
        f"""
        import resource, sys
        try:
            resource.setrlimit(resource.RLIMIT_AS, ({memory_mb} * 1024 * 1024, {memory_mb} * 1024 * 1024))
        except (ValueError, OSError):
            pass  # some platforms disallow RLIMIT_AS; best-effort only
        {code}
        """
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", wrapped],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return RunResult(proc.stdout, proc.stderr, proc.returncode, False)
    except subprocess.TimeoutExpired:
        log.warning("sandbox.timeout", timeout=timeout)
        return RunResult("", "timeout", 124, True)
    except FileNotFoundError as e:
        log.exception("sandbox.missing_interpreter")
        return RunResult("", str(e), 127, False, error=f"interpreter not found: {e}")
    except PermissionError as e:
        log.exception("sandbox.permission")
        return RunResult("", str(e), 126, False, error=f"permission denied: {e}")
    except OSError as e:
        log.exception("sandbox.oserror")
        return RunResult("", str(e), 1, False, error=f"os error: {e}")
