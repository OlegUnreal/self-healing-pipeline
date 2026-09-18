"""Wiring tests: memory and the trained classifier are accelerators, never the driver.

Everything here asserts one of two things: what the planner is shown when a store is
attached, and what the store is told when a run ends. None of it may move `report.success`.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from self_healing.agent import heal, heal_with_tools
from self_healing.config import Settings
from self_healing.corpus import REPAIR_EXAMPLES
from self_healing.demo_tools import BROKEN, FIX, TEST, scripted_planner
from self_healing.graph import GraphContext, heal_with_graph, initial_state, observe_node
from self_healing.memory import RepairMemory, open_memory
from self_healing.sandbox import RunResult
from self_healing.tools import Workspace, default_registry
from self_healing.verifier import classify_output, classify_with_ml, verify

BAD = "def add(a, b):\n    return a - b\n"
ADD_TEST = "from add import add\nassert add(2, 3) == 5\n"
GOOD_DIFF = (
    "--- a/add.py\n+++ b/add.py\n@@ -1,2 +1,2 @@\n"
    " def add(a, b):\n-    return a - b\n+    return a + b\n"
)
WRONG_DIFF = (
    "--- a/add.py\n+++ b/add.py\n@@ -1,2 +1,2 @@\n"
    " def add(a, b):\n-    return a - b\n+    return b - a\n"
)
TB = 'Traceback:\n  File "svc/client.py", line 12\nAssertionError: expected 5\n'


@pytest.fixture
def mem():
    with RepairMemory(":memory:", backend="hashing", dim=48) as memory:
        yield memory


def broken_add(root):
    root.mkdir(parents=True, exist_ok=True)
    src = root / "add.py"
    src.write_text(BAD, encoding="utf-8")
    return src


class FakeClassifier:
    """Stand-in for `TracebackClassifier`: only `predict` / `predict_proba` are seams."""

    def __init__(self, label="import_error", prob=0.9):
        self.label, self.prob = label, prob

    def predict(self, text):
        return self.label

    def predict_proba(self, texts):
        return np.array([[self.prob, 1.0 - self.prob] for _ in texts])


class ExplodingClassifier:
    def predict(self, text):
        raise RuntimeError("artifact was written by a different sklearn")


def rows(memory):
    return memory.conn.execute("SELECT diff, meta, success, failure_class FROM repairs ORDER BY ts").fetchall()


# ---------------------------------------------------------------- diff loop
def test_an_empty_store_leaves_the_proposer_prompt_untouched(tmp_path, mem):
    plain, wired = broken_add(tmp_path / "plain"), broken_add(tmp_path / "wired")
    seen: list[str] = []
    propose = lambda tb: (seen.append(tb), "")[1]  # noqa: E731
    heal(plain, ADD_TEST, propose, max_attempts=1)
    heal(wired, ADD_TEST, propose, max_attempts=1, memory=mem)
    assert seen[0] == seen[1]
    assert seen[0].startswith("[class=assertion_failure]")
    assert len(mem) == 0


def test_recall_hints_are_appended_after_the_traceback(tmp_path, mem):
    src = broken_add(tmp_path)
    red = verify(ADD_TEST, cwd=tmp_path)
    mem.record(red.stderr, GOOD_DIFF, failure_class=red.error, success=True)
    seen: list[str] = []
    heal(src, ADD_TEST, lambda tb: (seen.append(tb), "")[1], max_attempts=1, memory=mem)
    assert seen[0].startswith(red.stderr)
    assert "Similar past repairs" in seen[0]
    assert "return a + b" in seen[0]


def test_a_green_patch_is_learnt_even_when_the_budget_runs_out(tmp_path, mem):
    src = broken_add(tmp_path)
    report = heal(src, ADD_TEST, lambda _tb: GOOD_DIFF, max_attempts=1, memory=mem)
    assert report.success is False
    assert "return a + b" in report.final_source
    row = rows(mem)[0]
    assert int(row["success"]) == 1
    assert row["failure_class"] == "assertion_failure"
    assert json.loads(row["meta"])["loop"] == "diff"
    assert mem.stats()["success_rate"] == 1.0


def test_a_patch_that_stays_red_is_learnt_as_a_loss(tmp_path, mem):
    src = broken_add(tmp_path)
    report = heal(src, ADD_TEST, lambda _tb: WRONG_DIFF, max_attempts=1, memory=mem)
    assert report.success is False
    assert int(rows(mem)[0]["success"]) == 0


def test_verdicts_repeat_but_memory_never_changes_them(tmp_path, mem):
    plain, wired = broken_add(tmp_path / "plain"), broken_add(tmp_path / "wired")
    calls = {"n": 0}

    def proposer(tb):
        calls["n"] += 1
        return "" if calls["n"] % 2 == 0 else WRONG_DIFF

    plain_report = heal(plain, ADD_TEST, proposer, max_attempts=3)
    calls["n"] = 0
    wired_report = heal(wired, ADD_TEST, proposer, max_attempts=3, memory=mem)
    assert [(a.passed, a.diff_hash, a.error) for a in plain_report] == [
        (a.passed, a.diff_hash, a.error) for a in wired_report
    ]
    assert plain_report.success == wired_report.success
    got = rows(mem)
    assert [int(r["success"]) for r in got] == [0] * len(got)
    assert sum(json.loads(r["meta"]).get("seeded", 0) for r in got) == 0


def test_a_classifier_reaches_the_verifier_through_heal(tmp_path, mem):
    src = broken_add(tmp_path)
    heal(src, ADD_TEST, lambda _tb: GOOD_DIFF, max_attempts=1, memory=mem,
         classifier=FakeClassifier("import_error"))
    assert rows(mem)[0]["failure_class"] == "import_error"


# ---------------------------------------------------------------- tool loop
def test_tool_loop_records_one_net_diff_episode(tmp_path):
    src = tmp_path / "add.py"
    src.write_text(BROKEN, encoding="utf-8")
    with RepairMemory(":memory:", backend="hashing", dim=48) as memory:
        report = heal_with_tools(src, TEST, scripted_planner, max_steps=8, memory=memory)
        assert report.success is True
        assert len(memory) == 1
        row = rows(memory)[0]
        assert "a/add.py" in row["diff"] and "+    return a + b" in row["diff"]
        assert json.loads(row["meta"])["loop"] == "tools"


def test_workspace_memory_wins_over_the_argument(tmp_path):
    src = tmp_path / "add.py"
    src.write_text(BROKEN, encoding="utf-8")
    ws = Workspace(tmp_path)
    ours, theirs = RepairMemory(":memory:", backend="hashing", dim=48), RepairMemory(
        ":memory:", backend="hashing", dim=48
    )
    ws.memory = ours
    report = heal_with_tools(src, TEST, scripted_planner, max_steps=8, workspace=ws, memory=theirs)
    assert report.success is True
    assert len(ours) == 1 and len(theirs) == 0
    ours.close()
    theirs.close()


def test_recall_is_injected_exactly_once_as_a_user_turn(tmp_path):
    src = tmp_path / "add.py"
    src.write_text(BROKEN, encoding="utf-8")
    red = verify(TEST, cwd=tmp_path)
    with RepairMemory(":memory:", backend="hashing", dim=48) as memory:
        memory.record(red.stderr, FIX, failure_class=red.error, success=True)
        roles: list[list[str]] = []

        def planner(messages, schemas):
            roles.append([m["role"] for m in messages])
            return scripted_planner(messages, schemas)

        report = heal_with_tools(src, TEST, planner, max_steps=8, memory=memory)
        assert report.success is True
    assert roles[0].count("user") == 1
    assert max(r.count("user") for r in roles) == 2


def test_top_k_zero_skips_injection_but_still_learns(tmp_path):
    src = tmp_path / "add.py"
    src.write_text(BROKEN, encoding="utf-8")
    red = verify(TEST, cwd=tmp_path)
    with RepairMemory(":memory:", backend="hashing", dim=48) as memory:
        memory.record(red.stderr, FIX, failure_class=red.error, success=True)
        roles: list[list[str]] = []

        def planner(messages, schemas):
            roles.append([m["role"] for m in messages])
            return scripted_planner(messages, schemas)

        report = heal_with_tools(src, TEST, planner, max_steps=8, memory=memory, memory_top_k=0)
        assert report.success is True
        assert all(r.count("user") == 1 for r in roles)
        assert len(memory) == 1


def test_memory_adds_a_line_to_the_system_prompt(tmp_path):
    src = tmp_path / "add.py"
    src.write_text(BROKEN, encoding="utf-8")
    captured: list[str] = []

    def planner(messages, schemas):
        if [m["role"] for m in messages] == ["system", "user"]:
            captured.append(messages[0]["content"])
        return scripted_planner(messages, schemas)

    with RepairMemory(":memory:", backend="hashing", dim=48) as memory:
        heal_with_tools(src, TEST, planner, max_steps=8)
        src.write_text(BROKEN, encoding="utf-8")
        heal_with_tools(src, TEST, planner, max_steps=8, memory=memory)
    assert "recall_repairs" not in captured[0]
    assert captured[1].startswith(captured[0])
    assert "recall_repairs" in captured[1]


# ---------------------------------------------------------------- tools
def test_recall_repairs_tool_says_so_when_disabled(tmp_path):
    result = default_registry().call(Workspace(tmp_path), "recall_repairs", {"query": TB})
    assert result.ok is False
    assert "SHP_USE_ML" in result.error


def test_recall_repairs_tool_exposes_its_scoring(tmp_path, mem):
    mem.record(TB, GOOD_DIFF, failure_class="assertion_failure", success=True)
    ws = Workspace(tmp_path)
    ws.memory = mem
    result = default_registry().call(ws, "recall_repairs", {"query": TB, "k": 2})
    assert result.ok is True
    assert result.data["count"] >= 1
    hit = result.data["repairs"][0]
    assert {"class", "similarity", "prior", "score", "green", "attempts", "diff"} <= set(hit)
    assert hit["green"] is True and hit["similarity"] > 0.9


def test_recall_repairs_survives_a_broken_store(tmp_path):
    ws = Workspace(tmp_path)
    ws.memory = object()
    result = default_registry().call(ws, "recall_repairs", {"query": TB})
    assert result.ok is False
    assert result.error.startswith("recall failed:")


def test_classify_failure_keeps_the_rules_when_no_model_is_attached(tmp_path):
    result = default_registry().call(Workspace(tmp_path), "classify_failure", {"output": TB})
    assert result.data["backend"] == "rules"
    assert result.data["class"] == classify_output(RunResult(TB, "", 1, False))


def test_classify_failure_uses_the_injected_classifier(tmp_path):
    ws = Workspace(tmp_path)
    ws.classifier = FakeClassifier("import_error")
    result = default_registry().call(ws, "classify_failure", {"output": TB})
    assert result.data["backend"] == "learned"
    assert result.data["class"] == "import_error"


# ---------------------------------------------------------------- classifier seam
def test_the_model_declines_where_it_has_no_standing():
    red = RunResult("", "", 1, False)
    assert classify_with_ml(red, FakeClassifier("import_error")) is None


def test_an_unsure_or_broken_model_defers_to_the_keyword_ladder():
    red = RunResult("AssertionError: nope", "", 1, False)
    assert classify_with_ml(red, ExplodingClassifier()) is None
    assert classify_with_ml(red, FakeClassifier("ok")) is None
    assert verify('assert 1 == 2, "nope"', classifier=ExplodingClassifier()).error == "assertion_failure"


# ---------------------------------------------------------------- graph
def test_graph_injects_recalled_hints_once(tmp_path, mem):
    src = tmp_path / "add.py"
    src.write_text(BROKEN, encoding="utf-8")
    red = verify(TEST, cwd=tmp_path)
    mem.record(red.stderr, FIX, failure_class=red.error, success=True)
    ctx = GraphContext(
        source=src,
        test=TEST,
        planner=lambda *_: {},
        workspace=Workspace(tmp_path),
        registry=default_registry(),
        memory=mem,
    )
    state = initial_state(src, TEST, max_steps=3)
    first = observe_node(ctx, state)
    assert first["episode_traceback"] == red.stderr
    assert first["episode_class"] == red.error
    assert [m for m in first["messages"] if m["role"] == "user"][-1]["content"].startswith("Similar past repairs")
    state.update(first)
    second = observe_node(ctx, state)
    assert "episode_traceback" not in second and "messages" not in second


def test_graph_learns_the_episode_when_the_report_is_built(tmp_path, mem):
    src = tmp_path / "add.py"
    src.write_text(BROKEN, encoding="utf-8")
    report = heal_with_graph(src, TEST, scripted_planner, max_steps=8, memory=mem)
    assert report.success is True
    row = rows(mem)[0]
    assert json.loads(row["meta"])["loop"] == "graph"
    assert "+    return a + b" in row["diff"]


def test_a_green_workspace_never_opens_an_episode(tmp_path, mem):
    src = tmp_path / "add.py"
    src.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    ctx = GraphContext(
        source=src,
        test=TEST,
        planner=lambda *_: {},
        workspace=Workspace(tmp_path),
        registry=default_registry(),
        memory=mem,
    )
    update = observe_node(ctx, initial_state(src, TEST, max_steps=3))
    assert update["passed"] is True
    assert "episode_traceback" not in update
    assert len(mem) == 0


def test_without_memory_the_graph_never_pays_for_net_diff(tmp_path):
    src = tmp_path / "add.py"
    src.write_text(BROKEN, encoding="utf-8")
    ws = Workspace(tmp_path)

    def boom():
        raise AssertionError("net_diff is episode bookkeeping, not a default-path cost")

    ws.net_diff = boom
    report = heal_with_graph(src, TEST, scripted_planner, max_steps=8, workspace=ws)
    assert report.success is True


# ---------------------------------------------------------------- settings
def test_open_memory_is_off_unless_asked_for(tmp_path):
    path = tmp_path / "memory.sqlite3"
    assert open_memory(Settings(use_ml=False, memory_path=str(path))) is None
    assert not path.exists()


def test_open_memory_seeds_once_and_reuses_the_file(tmp_path):
    settings = Settings(use_ml=True, memory_path=str(tmp_path / "memory.sqlite3"))
    first = open_memory(settings)
    assert first is not None and 0 < len(first) <= len(REPAIR_EXAMPLES)
    first_count = len(first)
    first.close()
    again = open_memory(Settings(use_ml=True, memory_path=settings.memory_path))
    assert len(again) == first_count
    again.close()


def test_open_memory_gives_up_quietly_on_an_unusable_path(tmp_path):
    assert open_memory(Settings(use_ml=True, memory_path=str(tmp_path))) is None
