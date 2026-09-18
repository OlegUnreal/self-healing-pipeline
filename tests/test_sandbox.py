from self_healing.sandbox import run_code


def test_success():
    r = run_code("print(1+1)")
    assert r.exit_code == 0
    assert "2" in r.stdout
    assert r.error == ""


def test_failure():
    r = run_code("raise ValueError('boom')")
    assert r.exit_code != 0
    assert "boom" in r.stderr


def test_timeout_classified():
    r = run_code("import time; time.sleep(10)", timeout=0.1)
    assert r.timed_out is True
    assert r.exit_code == 124


def test_never_raises_on_host_error(monkeypatch):
    import subprocess
    real_run = subprocess.run

    def boom(*_a, **_k):
        raise PermissionError("nope")

    monkeypatch.setattr(subprocess, "run", boom)
    r = run_code("print(1)")
    assert r.exit_code == 126
    assert "permission denied" in r.error
    subprocess.run = real_run


def test_memory_cap_refuses_a_runaway_allocation():
    """`resource.setrlimit` has no Windows twin, so this fails loudly on whichever
    platform stops enforcing the cap - silently running uncapped was the bug."""
    probe = "print('alive'); x = bytearray(1024 * 1024 * 1024); print('allocated', len(x))"
    r = run_code(probe, memory_mb=256)
    assert "alive" in r.stdout, f"child never started: {r.stderr[-300:]}"
    assert "allocated" not in r.stdout
    assert "MemoryError" in r.stderr


def test_memory_cap_is_inherited_by_grandchildren():
    probe = (
        "import subprocess, sys\n"
        "p = subprocess.run([sys.executable, '-c', 'x = bytearray(1024*1024*1024)'],\n"
        "                   capture_output=True, text=True)\n"
        "print('exit', p.returncode)\n"
        "print(p.stderr.strip().splitlines()[-1] if p.stderr.strip() else 'no error')\n"
    )
    r = run_code(probe, memory_mb=256)
    assert r.exit_code == 0, r.stderr[-300:]
    assert "MemoryError" in r.stdout
