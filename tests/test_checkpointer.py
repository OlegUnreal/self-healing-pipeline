from self_healing.graph import make_checkpointer


def test_make_checkpointer_without_extra_is_none_or_memory():
    cp = make_checkpointer(None)
    assert cp is None or type(cp).__name__ in {"InMemorySaver", "MemorySaver"}
