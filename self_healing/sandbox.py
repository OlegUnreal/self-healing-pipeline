"""Sandbox executor: runs untrusted Python in an isolated subprocess.

Isolation has to be platform-specific because the two OSes expose different
primitives:

* POSIX - `preexec_fn` sets `RLIMIT_AS` and `os.setsid()` in the forked child before
  it execs, so the cap is in place as the interpreter starts and is inherited by
  grandchildren.
* Windows - `preexec_fn`/`start_new_session` do not exist (passing them is a
  `ValueError`, not a silent no-op) and there is no `resource` module. The equivalent
  is a Job Object with `JOB_OBJECT_LIMIT_PROCESS_MEMORY`. A process may assign itself
  to a job, so the cap is installed by a `usercustomize.py` bootstrap that is put on
  `PYTHONPATH`; it also covers grandchildren, which mirrors the rlimit behaviour.

Both paths fail soft: a cap the OS refuses to install must not turn a passing repair
into a harness error.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

from . import logging as logmod

log = logmod.get_logger(__name__)

IS_WINDOWS = os.name == "nt"
_MEM_ENV = "SHP_SANDBOX_MEM_MB"


@dataclass
class RunResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    error: str = ""


def _preexec_limits(memory_mb: int) -> None:
    """Cap memory inside the child. Runs in preexec_fn."""
    try:
        import resource

        cap = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    except (ValueError, OSError, ImportError):
        pass
    try:
        os.setsid()
    except (AttributeError, OSError):
        pass


_JOB_LIMIT_PROCESS_MEMORY = 0x100
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9

_BOOTSTRAP = '''"""Interpreter-startup hook installed by self_healing.sandbox on Windows.

site.py imports this module before any user code runs, which is the earliest hook a
parent process has: the child is placed in a memory-capped job object from inside.
"""
import os
import sys


def _cap_process_memory():
    mb = int(os.environ.get("SHP_SANDBOX_MEM_MB", "0") or 0)
    if mb <= 0 or sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    class _Basic(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class _Io(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _Extended(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _Basic),
            ("IoInfo", _Io),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return
    info = _Extended()
    info.BasicLimitInformation.LimitFlags = %d
    info.ProcessMemoryLimit = mb * 1024 * 1024
    if not k32.SetInformationJobObject(
            job, %d, ctypes.byref(info), ctypes.sizeof(info)):
        return
    # Already in a non-nestable job? Then the cap is unavailable and we run uncapped,
    # exactly like the POSIX path does when setrlimit is refused.
    k32.AssignProcessToJobObject(job, k32.GetCurrentProcess())


try:
    _cap_process_memory()
except Exception:
    pass
''' % (_JOB_LIMIT_PROCESS_MEMORY, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION)

_bootstrap_dir: str | None = None
_bootstrap_failed = False


def _bootstrap_path() -> str | None:
    """Materialise the cap hook once per process; None when it cannot be written."""
    global _bootstrap_dir, _bootstrap_failed
    if _bootstrap_dir is not None or _bootstrap_failed:
        return _bootstrap_dir
    from tempfile import mkdtemp

    try:
        directory = mkdtemp(prefix="shp-sandbox-")
        with open(os.path.join(directory, "usercustomize.py"), "w", encoding="ascii") as fh:
            fh.write(_BOOTSTRAP)
    except OSError as exc:
        log.warning("sandbox.bootstrap_unavailable", error=str(exc))
        _bootstrap_failed = True
        return None
    _bootstrap_dir = directory
    return directory


def _child_env(memory_mb: int) -> dict[str, str]:
    env = dict(os.environ)
    directory = _bootstrap_path()
    if directory is None:
        return env
    env[_MEM_ENV] = str(int(memory_mb))
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = directory + (os.pathsep + existing if existing else "")
    return env


def run_code(
    code: str,
    timeout: float = 5.0,
    memory_mb: int = 256,
    cwd: str | Path | None = None,
) -> RunResult:
    """Execute `code` in a fresh interpreter.

    `cwd` is the working directory of the child so tests can `import` modules
    that live next to the file being repaired.
    """
    wrapped = textwrap.dedent(code).lstrip("\n")
    workdir = str(cwd) if cwd is not None else None
    if IS_WINDOWS:
        # A new process group is the closest available stand-in for setsid(): it stops
        # Ctrl+C in the parent console from taking the sandbox down with it.
        isolation = {
            "env": _child_env(memory_mb),
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP,
        }
    else:
        isolation = {"preexec_fn": lambda: _preexec_limits(memory_mb), "start_new_session": True}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", wrapped],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir,
            **isolation,
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
