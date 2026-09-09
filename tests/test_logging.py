import json

from self_healing.logging import EventLog, get_logger


def test_event_log_roundtrip():
    ev = EventLog()
    ev.add("x", a=1)
    ev.add("y", b="z")
    data = json.loads(ev.as_json())
    assert data[0]["event"] == "x" and data[0]["a"] == 1
    assert data[1]["event"] == "y"


def test_logger_is_json(capsys):
    lg = get_logger("test_json_logger")
    lg.info("hello", foo="bar")
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["event"] == "hello"
    assert payload["foo"] == "bar"
    assert "level" in payload
