"""Episodic repair memory: store, blend similarity with a learned prior, reuse."""
from __future__ import annotations

import json

import pytest

from self_healing.corpus import REPAIR_EXAMPLES
from self_healing.memory import RepairMemory, RepairRecord, seed_memory
from self_healing.vector_store import FlatIndex, IvfIndex

TB = """Traceback (most recent call last):
  File "svc/client.py", line 12, in fetch
    return session.get(url, timeout=timeout)
TimeoutError: The read operation timed out
"""
DIFF = "--- a/svc/client.py\n+++ b/svc/client.py\n+timeout = 60\n"


@pytest.fixture
def mem():
    with RepairMemory(":memory:", backend="hashing", dim=48) as memory:
        yield memory


def test_record_then_recall_finds_the_same_failure(mem):
    mem.record(TB, DIFF, failure_class="timeout", success=True)
    hits = mem.recall(TB.replace("read operation timed out", "read operation timed out again"))
    assert hits and hits[0].record.failure_class == "timeout"
    assert hits[0].similarity > 0.5
    assert hits[0].as_prompt().startswith("[")


def test_empty_memory_recalls_nothing(mem):
    assert mem.recall("whatever failed") == []
    assert mem.prompt_block("whatever failed") == ""
    assert mem.stats()["index"] == "empty"


def test_identical_repair_folds_into_one_row_and_accumulates_attempts(mem):
    mem.record(TB, DIFF, failure_class="timeout", success=False, attempts=1)
    mem.record(TB, DIFF, failure_class="timeout", success=True, attempts=2)
    assert len(mem) == 1
    row = mem.conn.execute("SELECT attempts, success FROM repairs").fetchone()
    assert int(row["attempts"]) == 3 and int(row["success"]) == 1


def test_success_prior_reshuffles_equal_similarity_hits(mem):
    other = TB.replace("svc/client.py", "svc/worker.py")
    mem.record(TB, DIFF, failure_class="timeout", success=True)
    mem.record(other, DIFF, failure_class="timeout", success=False)
    for _ in range(6):
        mem.bump("timeout", success=True)
    winning = mem.recall(TB, k=1, weights={"similarity": 0.5, "prior": 0.5})[0]
    assert winning.record.traceback == TB
    assert winning.prior > 0.5


def test_failure_class_filter_and_similarity_floor(mem):
    mem.record(TB, DIFF, failure_class="timeout", success=True)
    assert mem.recall(TB, failure_class="name_error") == []
    assert mem.recall("an unrelated sentence about cooking", min_similarity=0.9) == []


def test_stats_report_shape(mem):
    assert seed_memory(mem) == len(REPAIR_EXAMPLES)
    stats = mem.stats()
    assert stats["records"] == len(REPAIR_EXAMPLES)
    assert stats["embedder"] == "hashing"
    assert 0.0 <= stats["success_rate"] <= 1.0
    assert set(stats["by_class"]) <= {e["failure_class"] for e in REPAIR_EXAMPLES}


def test_export_jsonl_is_machine_readable(mem):
    mem.record(TB, DIFF, failure_class="timeout", success=True, run_id="r1")
    rows = [json.loads(line) for line in mem.export_jsonl().splitlines()]
    assert rows[0]["meta"] == {"run_id": "r1"}
    assert rows[0]["failure_class"] == "timeout"


def test_index_switches_to_ivf_above_threshold(tmp_path):
    path = tmp_path / "memory.sqlite3"
    with RepairMemory(path, backend="hashing", dim=32, ivf_threshold=4) as memory:
        for i in range(8):
            memory.record(
                f'Traceback: File "m{i}.py", line {i}\n{name_cls(i)}Error: variant {i} blew up',
                f"--- a/m{i}.py\n+++ b/m{i}.py\n+fixed {i}\n",
                failure_class="runtime_error",
                success=bool(i % 2),
            )
        assert isinstance(memory.index, IvfIndex)
        assert memory.recall('File "m3.py" RuntimeError variant 3 blew up', k=1)
    with RepairMemory(path, backend="hashing", dim=32) as reopened:
        assert isinstance(reopened.index, FlatIndex)
        assert len(reopened) == 8
        assert reopened.recall('File "m3.py" RuntimeError variant 3 blew up', k=2)


def name_cls(i: int) -> str:
    return ["Value", "Type", "Name", "Runtime"][i % 4]


def test_reopening_with_a_different_embedder_self_heals(tmp_path):
    path = tmp_path / "m.sqlite3"
    with RepairMemory(path, backend="hashing", dim=32) as memory:
        memory.record(TB, DIFF, failure_class="timeout", success=True)
    with RepairMemory(path, backend="hashing", dim=96) as reopened:
        hits = reopened.recall(TB, k=1)
        assert len(hits) == 1 and hits[0].record.failure_class == "timeout"
        stored = reopened.conn.execute("SELECT embedder, length(embedding) n FROM repairs").fetchone()
        assert stored["embedder"] == "hashing"
        assert stored["n"] == reopened.stats()["dim"] * 4
    with RepairMemory(path, backend="hashing", dim=96) as again:
        assert len(again.recall(TB, k=1)) == 1


def test_fit_embedder_skips_small_corpuses_and_refits_large_ones(mem):
    assert "skipped" in mem.fit_embedder(["short one", "short two"])
    corpus = [f'File "a{i}.py", line {i}\n{i%3} Error: distinct message number {i}' for i in range(20)]
    for i, text in enumerate(corpus):
        mem.record(text, f"+patch {i}\n", failure_class="runtime_error", success=True)
    report = mem.fit_embedder(corpus)
    assert "refit tfidf-lsa" in report
    assert mem.backend_name == "tfidf-lsa"
    assert mem.recall(corpus[3], k=2)


def test_add_accepts_explicit_record_objects(mem):
    record = RepairRecord(TB, DIFF, "timeout", True)
    mem.add(record)
    assert mem.recall(TB, k=1)[0].record.id == record.id
    assert record.tb_hash == RepairRecord(TB, "", "", True).tb_hash
