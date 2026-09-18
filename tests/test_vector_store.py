"""Vector index tests: exactness for flat, *measured* approximation for IVF.

`test_ivf_recall_is_reported_and_gated` is the regression guard -- if a future change
to the k-means routing or to `n_probe` silently degrades retrieval, this fails loudly
instead of shipping a memory that returns the wrong past repairs.
"""
from __future__ import annotations

import numpy as np
import pytest

from self_healing.vector_store import FlatIndex, IvfIndex, mmr, recall_at_k


def _unit(rows):
    m = np.asarray(rows, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def corpus():
    rng = np.random.default_rng(11)
    centers = _unit(rng.normal(size=(8, 16)))
    ids, vectors = [], []
    for c in range(centers.shape[0]):
        for r in range(12):
            v = centers[c] + 0.35 * rng.normal(size=16)
            ids.append(f"doc-{c}-{r}")
            vectors.append(v)
    return ids, _unit(vectors)


def test_flat_index_empty_search_returns_nothing():
    assert FlatIndex().search([1.0, 0.0], k=3) == []


def test_flat_index_ranks_by_cosine_and_rejects_dim_drift():
    index = FlatIndex()
    index.add([1.0, 0.0], "x")
    index.add([0.0, 1.0], "y")
    hits = index.search([0.9, 0.1], k=2)
    assert hits[0][0] == "x"
    assert hits[0][1] > hits[1][1]
    assert len(index) == 2 and index.dim == 2
    with pytest.raises(ValueError, match="dim mismatch"):
        index.add([1.0, 2.0, 3.0], "z")


def test_recall_at_k_handles_empty_golden():
    assert recall_at_k([], ["a"]) == 1.0
    assert recall_at_k(["a", "b"], ["a", "b", "c"]) == 1.0
    assert recall_at_k(["a", "b"], ["b", "c", "a"]) == 0.5
    assert recall_at_k(["a", "b"], ["c", "d"]) == 0.0


def test_ivf_full_probe_matches_flat_exactly(corpus):
    ids, vectors = corpus
    ivf = IvfIndex(ids, vectors, n_lists=8, n_probe=8)
    flat = FlatIndex(ids, vectors)
    for row in vectors[:10]:
        assert [i for i, _ in ivf.search(row, 5)] == [i for i, _ in flat.search(row, 5)]


def test_ivf_recall_is_reported_and_gated(corpus):
    ids, vectors = corpus
    ivf = IvfIndex(ids, vectors, n_lists=8, n_probe=4)
    queries = _unit(np.asarray(vectors[5:35], dtype=np.float32) + 0.05)
    metrics = ivf.benchmark(queries, k=5)
    assert set(metrics) == {"recall@5"}
    assert metrics["recall@5"] >= 0.90, f"IVF recall@5 regressed: {metrics}"


def test_ivf_reports_posting_balance(corpus):
    ids, vectors = corpus
    balance = IvfIndex(ids, vectors, n_lists=8, n_probe=2).balance()
    assert balance["docs"] == len(ids)
    assert balance["coverage"] == 1.0
    assert 0 < balance["largest_list"] <= len(ids)


def test_ivf_degenerates_to_exact_search_on_a_tiny_corpus():
    vectors = _unit([[1, 0, 1], [0, 1, 2]])
    ivf = IvfIndex(["a", "b"], vectors, n_lists=32)
    assert ivf.n_lists == 2
    assert ivf.centroids.shape[0] == 1
    assert len(ivf.candidates([1, 0, 1])) == 2
    flat = FlatIndex(["a", "b"], vectors)
    assert [i for i, _ in ivf.search([1, 0, 1], k=2)] == [i for i, _ in flat.search([1, 0, 1], k=2)]


def test_mmr_trades_a_little_relevance_for_diversity():
    query = [1.0, 0.0]
    dupes = [_unit([[1.0, 0.0]])[0] for _ in range(4)]
    spread = _unit([[0.9063, 0.4226], [0.9063, -0.4226], [0.4226, 0.9063]])
    items = [(f"dup{i}", dupes[i]) for i in range(4)] + [(f"far{i}", spread[i]) for i in range(3)]
    picked = [i for i, _ in mmr(query, items, k=4, lambda_mult=0.1)]
    assert sum(1 for p in picked if p.startswith("dup")) == 1
    assert len({*picked}) == 4
    pure_cosine = [i for i, _ in sorted(((i, float(v @ np.asarray(query))) for i, v in items), key=lambda kv: -kv[1])[:4]]
    assert all(p.startswith("dup") for p in pure_cosine), "baseline should be the boring all-duplicates ranking"


def test_mmr_on_empty_input():
    assert mmr([1.0, 0.0], [], k=3) == []
