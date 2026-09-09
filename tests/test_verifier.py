from self_healing.verifier import _classify, verify


def test_classify_syntax():
    r = type("R", (), {"timed_out": False, "stderr": "SyntaxError: bad", "stdout": "", "exit_code": 1})()
    assert _classify(r) == "syntax_error"


def test_classify_assertion():
    r = type("R", (), {"timed_out": False, "stderr": "AssertionError: x", "stdout": "FAIL", "exit_code": 1})()
    assert _classify(r) == "assertion_failure"


def test_classify_timeout():
    r = type("R", (), {"timed_out": True, "stderr": "", "stdout": "", "exit_code": 124})()
    assert _classify(r) == "timeout"


def test_verify_tags_stderr():
    r = verify('raise AssertionError("x")')
    assert r.exit_code != 0
    assert r.stderr.startswith("[class=assertion_failure]")
