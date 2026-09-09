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
