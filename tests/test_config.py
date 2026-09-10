from self_healing.config import Settings, load_settings


def test_defaults(monkeypatch):
    monkeypatch.delenv("SHP_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    s = load_settings()
    assert s.model == "gpt-4o-mini"
    assert s.max_attempts == 5


def test_env_override(monkeypatch):
    monkeypatch.setenv("SHP_MODEL", "gpt-4o")
    monkeypatch.setenv("SHP_MAX_ATTEMPTS", "9")
    s = load_settings()
    assert s.model == "gpt-4o"
    assert s.max_attempts == 9
    assert isinstance(s, Settings)
