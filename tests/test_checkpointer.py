import pytest

from self_healing.graph import CheckpointerUnavailable, make_checkpointer


def test_make_checkpointer_without_extra_is_none_or_memory():
    cp = make_checkpointer(None, required=False)
    assert cp is None or type(cp).__name__ in {"InMemorySaver", "MemorySaver"}


def test_required_sqlite_raises_when_unavailable(tmp_path, monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "langgraph.checkpoint.sqlite", None)
    with pytest.raises(CheckpointerUnavailable):
        make_checkpointer(tmp_path / "heal.sqlite", required=True)
