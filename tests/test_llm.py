from self_healing.llm import _strip_fences, make_openai_planner, make_openai_proposer


class _Msg:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, content, tool_calls=None):
        self.message = _Msg(content, tool_calls)


class _Resp:
    def __init__(self, content, tool_calls=None):
        self.choices = [_Choice(content, tool_calls)]


class _Stub:
    def __init__(self, content="--- a/x.py\n+++ b/x.py\n"):
        self.content = content
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return _Resp(self.content)


def test_strip_fences():
    raw = "```diff\n--- a/x\n+++ b/x\n```"
    assert _strip_fences(raw).startswith("--- a/x")


def test_proposer_uses_injected_client():
    stub = _Stub("--- a/add.py\n+++ b/add.py\n")
    propose = make_openai_proposer(client=stub)
    assert "add.py" in propose("boom")
    assert stub.calls == 1


def test_planner_parses_tool_calls():
    class Fn:
        name = "read_file"
        arguments = '{"path": "add.py"}'

    class Call:
        function = Fn()

    class Client:
        def create(self, **kwargs):
            return _Resp("", [Call()])

    planner = make_openai_planner(client=Client())
    out = planner([{"role": "user", "content": "fix it"}], [])
    assert out["tool_calls"][0]["name"] == "read_file"
    assert out["tool_calls"][0]["arguments"]["path"] == "add.py"
