"""Vector indexes: exact flat scan and an IVF ANN index with measured recall.

The point of keeping both is that the ANN index is only honest if we can report what it
costs: `IvfIndex.benchmark` recomputes exact neighbours and returns recall@k against the
flat baseline, so a regression in `n_probe` shows up as a failing test, not a silent
quality drop in production retrieval.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .embeddings import _normalise


def _as_matrix(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return _normalise(matrix)


def recall_at_k(golden: Iterable[str], retrieved: Iterable[str]) -> float:
    gold = list(golden)
    got = list(retrieved)
    if not gold:
        return 1.0
    hits = sum(1 for item in got[: len(gold)] if item in set(gold))
    return hits / len(gold)


@dataclass
class FlatIndex:
    """Exact cosine search over a dense float32 matrix."""

    ids: list[str] = field(default_factory=list)
    vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.size else 0

    def add(self, vector: Sequence[float], doc_id: str) -> None:
        row = _normalise(np.asarray(vector, dtype=np.float32).reshape(1, -1))
        if self.vectors.size and self.vectors.shape[1] != row.shape[1]:
            raise ValueError(f"dim mismatch: index={self.vectors.shape[1]} added={row.shape[1]}")
        self.vectors = row if not self.vectors.size else np.vstack([self.vectors, row])
        self.ids.append(doc_id)

    def search(self, query: Sequence[float], k: int = 5) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        q = _normalise(np.asarray(query, dtype=np.float32).reshape(1, -1))
        if q.shape[1] != self.dim:
            raise ValueError(f"query dim {q.shape[1]} != index dim {self.dim}")
        sims = (q @ self.vectors.T)[0]
        order = np.argsort(-sims, kind="stable")[: max(0, k)]
        return [(self.ids[i], float(sims[i])) for i in order]


def _lloyd_kmeans(matrix: np.ndarray, clusters: int, *, iters: int = 25, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = matrix.shape[0]
    centroids = matrix[rng.choice(n, size=clusters, replace=False)]
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        sims = matrix @ centroids.T
        new_labels = np.argmax(sims, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(clusters):
            members = matrix[labels == c]
            if members.size:
                centroids[c] = members.mean(axis=0)
        centroids = _normalise(centroids)
    return centroids, labels


@dataclass
class IvfIndex:
    """Inverted-file ANN: k-means centroids route queries to `n_probe` lists."""

    ids: list[str]
    vectors: np.ndarray
    n_lists: int = 0
    n_probe: int = 4
    centroids: np.ndarray = field(default=None, repr=False, init=False)  # type: ignore[assignment]
    postings: list[list[int]] = field(default_factory=list, repr=False, init=False)
    labels: np.ndarray = field(default=None, repr=False, init=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.vectors = _as_matrix(self.vectors)
        n = len(self.ids)
        if not self.n_lists:
            self.n_lists = int(max(1, min(32, np.sqrt(max(n, 1)))))
        self.build()

    def build(self, *, seed: int = 7) -> "IvfIndex":
        n = len(self.ids)
        lists = int(max(1, min(self.n_lists, n)))
        self.n_lists = lists
        if lists == 1 or n <= lists:
            self.centroids = self.vectors[:1] if n else np.zeros((1, self.vectors.shape[1]), dtype=np.float32)
            self.labels = np.zeros(n, dtype=np.int32)
        else:
            self.centroids, self.labels = _lloyd_kmeans(self.vectors, lists, seed=seed)
        self.postings = [[] for _ in range(self.centroids.shape[0])]
        for row, label in enumerate(self.labels):
            self.postings[int(label)].append(int(row))
        return self

    def _probe_lists(self, query: np.ndarray, n_probe: int) -> list[int]:
        sims = self.centroids @ query[0]
        return [int(i) for i in np.argsort(-sims, kind="stable")[: max(1, n_probe)]]

    def candidates(self, query: Sequence[float], n_probe: int | None = None) -> list[int]:
        q = _normalise(np.asarray(query, dtype=np.float32).reshape(1, -1))
        return [row for lst in self._probe_lists(q, n_probe or self.n_probe) for row in self.postings[lst]]

    def search(self, query: Sequence[float], k: int = 5, n_probe: int | None = None) -> list[tuple[str, float]]:
        rows = self.candidates(query, n_probe)
        if not rows:
            return []
        q = _normalise(np.asarray(query, dtype=np.float32).reshape(1, -1))
        sims = self.vectors[rows] @ q[0]
        order = np.argsort(-sims, kind="stable")[:k]
        return [(self.ids[rows[i]], float(sims[i])) for i in order]

    def balance(self) -> dict[str, float]:
        sizes = np.asarray([len(p) for p in self.postings], dtype=np.float32)
        return {
            "lists": int(self.n_lists),
            "docs": int(len(self.ids)),
            "empty_lists": int((sizes == 0).sum()),
            "largest_list": int(sizes.max()) if sizes.size else 0,
            "coverage": float((sizes > 0).mean()) if sizes.size else 0.0,
        }

    def benchmark(self, queries: Sequence[Sequence[float]], k: int = 5) -> dict[str, float]:
        """Exact-vs-approximate agreement; the number a reviewer should ask for."""
        flat = FlatIndex(list(self.ids), self.vectors)
        recalls = [recall_at_k([i for i, _ in flat.search(q, k)], [i for i, _ in self.search(q, k)]) for q in queries]
        return {f"recall@{k}": float(np.mean(recalls)) if recalls else 1.0}


def mmr(
    query: Sequence[float],
    items: Sequence[tuple[str, Sequence[float]]],
    k: int = 5,
    *,
    lambda_mult: float = 0.7,
) -> list[tuple[str, float]]:
    """Maximal marginal relevance: relevance against redundancy.

    Repair traces cluster hard (same traceback, different project), so pure cosine
    returns k copies of one fix and the planner learns nothing new.
    """
    if not items:
        return []
    ids = [i for i, _ in items]
    matrix = _as_matrix([v for _, v in items])
    q = _normalise(np.asarray(query, dtype=np.float32).reshape(1, -1))[0]
    relevance = matrix @ q
    selected: list[int] = []
    for _ in range(min(k, len(ids))):
        best, best_score = -1, -np.inf
        for idx in range(len(ids)):
            if idx in selected:
                continue
            if not selected:
                score = relevance[idx]
            else:
                redundancy = float(np.max(matrix[selected] @ matrix[idx]))
                score = lambda_mult * relevance[idx] - (1 - lambda_mult) * redundancy
            if score > best_score:
                best, best_score = idx, score
        if best < 0:
            break
        selected.append(best)
    return [(ids[i], float(relevance[i])) for i in selected]
