from self_healing.sandbox import run_code


def test_success():
    r = run_code("print(1+1)")
    assert r.exit_code == 0
    assert "2" in r.stdout


def test_failure():
    r = run_code("raise ValueError('boom')")
    assert r.exit_code != 0
    assert "boom" in r.stderr
