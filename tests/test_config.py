from self_healing.config import Settings, load_settings


def test_defaults(monkeypatch):
    monkeypatch.delenv("SHP_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SHP_USE_LANGGRAPH", raising=False)
    s = load_settings()
    assert s.model == "gpt-4o-mini"
    assert s.max_attempts == 5
    assert s.use_langgraph is False


def test_env_override(monkeypatch):
    monkeypatch.setenv("SHP_MODEL", "gpt-4o")
    monkeypatch.setenv("SHP_MAX_ATTEMPTS", "9")
    monkeypatch.setenv("SHP_USE_LANGGRAPH", "true")
    s = load_settings()
    assert s.model == "gpt-4o"
    assert s.max_attempts == 9
    assert s.use_langgraph is True
    assert isinstance(s, Settings)
