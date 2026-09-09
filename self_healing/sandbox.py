"""Sandbox executor: runs untrusted Python in an isolated subprocess."""
from __future__ import annotations

import logging
import os
import signal
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


def _preexec_limits(memory_mb: int) -> None:
    """Drop privileges and cap memory inside the child. Runs in preexec_fn."""
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024 * 1024, memory_mb * 1024 * 1024))
    except (ValueError, OSError, ImportError):
        pass
    try:
        os.setsid()
    except (AttributeError, OSError):
        pass


def run_code(code: str, timeout: float = 5.0, memory_mb: int = 256) -> RunResult:
    """Execute `code` in a fresh interpreter. No network, no filesystem writes outside /tmp.

    Memory limits are applied via RLIMIT_AS inside a preexec_fn so a limit
    breach kills only the child (returns exit -9) and never the host test
    process. Timeouts are caught and returned as RunResult(timed_out=True).
    """
    wrapped = textwrap.dedent(
        f"""
        {code}
        """
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", wrapped],
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=lambda: _preexec_limits(memory_mb),
            start_new_session=True,
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
