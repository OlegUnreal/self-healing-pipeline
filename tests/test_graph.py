from self_healing.demo_tools import BROKEN, FIX, TEST, scripted_planner
from self_healing.graph import (
    END,
    GraphContext,
    heal_with_graph,
    initial_state,
    iter_heal_graph,
    langgraph_available,
    observe_node,
    route_after_observe,
)
from self_healing.tools import Workspace, default_registry


def test_observe_green_routes_to_end(tmp_path):
    src = tmp_path / "add.py"
    src.write_text("def add(a, b):\n    return a + b\n")
    ctx = GraphContext(
        source=src,
        test=TEST,
        planner=lambda *_: {},
        workspace=Workspace(tmp_path),
        registry=default_registry(),
    )
    state = initial_state(src, TEST, max_steps=3)
    state.update(observe_node(ctx, state))
    assert state["passed"] is True
    assert route_after_observe(state) == END


def test_observe_fail_routes_to_plan(tmp_path):
    src = tmp_path / "add.py"
    src.write_text("def add(a, b):\n    return a - b\n")
    ctx = GraphContext(
        source=src,
        test=TEST,
        planner=lambda *_: {},
        workspace=Workspace(tmp_path),
        registry=default_registry(),
    )
    state = initial_state(src, TEST, max_steps=3)
    state.update(observe_node(ctx, state))
    assert state["passed"] is False
    assert route_after_observe(state) == "plan"


def test_budget_routes_to_escalate(tmp_path):
    src = tmp_path / "add.py"
    src.write_text("def add(a, b):\n    return a - b\n")
    state = initial_state(src, TEST, max_steps=0)
    state["passed"] = False
    assert route_after_observe(state) == "escalate"


def test_heal_with_graph_scripted(tmp_path):
    src = tmp_path / "add.py"
    src.write_text(BROKEN)
    report = heal_with_graph(src, TEST, scripted_planner, max_steps=8)
    assert report.success is True
    assert "return a + b" in report.final_source
    tools = [t["tool"] for t in report.tool_trace]
    assert "apply_patch" in tools
    assert "run_tests" in tools


def test_iter_heal_graph_emits_nodes(tmp_path):
    src = tmp_path / "add.py"
    src.write_text(BROKEN)
    ctx = GraphContext(
        source=src,
        test=TEST,
        planner=scripted_planner,
        workspace=Workspace(tmp_path),
        registry=default_registry(),
    )
    names = [name for name, _ in iter_heal_graph(ctx, initial_state(src, TEST, 8))]
    assert names[0] == "observe"
    assert "plan" in names
    assert "tools" in names
    assert names[-1] == "observe"


def test_mutation_rejected_without_approver(tmp_path):
    src = tmp_path / "add.py"
    src.write_text("def add(a, b):\n    return a - b\n")

    def planner(messages, schemas):
        return {
            "content": "patch",
            "tool_calls": [{"name": "apply_patch", "arguments": {"path": "add.py", "diff": FIX}}],
        }

    report = heal_with_graph(
        src,
        TEST,
        planner,
        max_steps=2,
        approve_mutations=True,
        mutation_approver=None,
    )
    assert report.success is False
    assert any(t["error"].startswith("mutation rejected") for t in report.tool_trace)
    assert "return a - b" in src.read_text()


def test_mutation_approved(tmp_path):
    src = tmp_path / "add.py"
    src.write_text(BROKEN)
    seen = {"n": 0}

    def planner(messages, schemas):
        seen["n"] += 1
        if seen["n"] == 1:
            return {
                "content": "patch",
                "tool_calls": [{"name": "apply_patch", "arguments": {"path": "add.py", "diff": FIX}}],
            }
        return {"content": "done", "tool_calls": [{"name": "finish", "arguments": {}}]}

    report = heal_with_graph(
        src,
        TEST,
        planner,
        max_steps=5,
        approve_mutations=True,
        mutation_approver=lambda _call: True,
    )
    assert report.success is True
    assert "return a + b" in src.read_text()


def test_langgraph_flag_is_boolean():
    assert langgraph_available() in {True, False}
