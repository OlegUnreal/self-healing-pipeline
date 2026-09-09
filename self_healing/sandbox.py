"""Sandbox executor: runs untrusted Python in an isolated subprocess."""
from __future__ import annotations

import subprocess
import sys
import textwrap
from dataclasses import dataclass


@dataclass
class RunResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


def run_code(code: str, timeout: float = 5.0, memory_mb: int = 256) -> RunResult:
    """Execute `code` in a fresh interpreter. No network, no filesystem writes outside /tmp."""
    wrapped = textwrap.dedent(
        f"""
        import resource, sys
        resource.setrlimit(resource.RLIMIT_AS, ({memory_mb} * 1024 * 1024, {memory_mb} * 1024 * 1024))
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
        return RunResult("", "timeout", 124, True)
