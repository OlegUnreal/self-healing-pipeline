"""Episodic repair memory: the same jail should not pay for the same mistake twice.

Every finished attempt is stored as (traceback, diff, outcome) and embedded. Recall blends
semantic similarity with a Laplace-smoothed success prior, so a fix that has worked four
times outranks a near-identical one that never did. Storage is SQLite; the vector index is
rebuilt lazily from the rows, which keeps the artifact a single portable file.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

# The core pipeline imports `observe` from this module, so the ml extra has to stay
# optional at import time: a stdlib-only install gets a memory that says "unavailable"
# rather than an ImportError in the middle of a repair run.
try:
    import numpy as np

    from .embeddings import HashingEmbedder, build_embedder
    from .vector_store import FlatIndex, IvfIndex, mmr

    _ML_IMPORT_ERROR = ""
except ImportError as exc:  # pragma: no cover - depends on the installed extra
    np = None
    HashingEmbedder = build_embedder = FlatIndex = IvfIndex = mmr = None
    _ML_IMPORT_ERROR = str(exc)

log = logging.getLogger("self_healing.memory")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS repairs (
    id            TEXT PRIMARY KEY,
    ts            REAL NOT NULL,
    failure_class TEXT NOT NULL,
    traceback     TEXT NOT NULL,
    diff          TEXT NOT NULL,
    success       INTEGER NOT NULL,
    attempts      INTEGER NOT NULL,
    tb_hash       TEXT NOT NULL,
    diff_hash     TEXT NOT NULL,
    embedder      TEXT NOT NULL,
    embedding     BLOB NOT NULL,
    meta          TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS repairs_class ON repairs(failure_class);
CREATE INDEX IF NOT EXISTS repairs_diff ON repairs(diff_hash);
CREATE TABLE IF NOT EXISTS counters (
    key TEXT PRIMARY KEY,
    hits INTEGER NOT NULL,
    wins INTEGER NOT NULL
);
"""

IVF_THRESHOLD = 256


@dataclass
class RepairRecord:
    traceback: str
    diff: str
    failure_class: str
    success: bool
    attempts: int = 1
    ts: float = field(default_factory=time.time)
    id: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = hashlib.sha1(f"{self.tb_hash}|{self.diff_hash}".encode()).hexdigest()[:16]

    @property
    def tb_hash(self) -> str:
        return hashlib.sha1(self.traceback.encode()).hexdigest()[:16]

    @property
    def diff_hash(self) -> str:
        return hashlib.sha1(self.diff.encode()).hexdigest()[:16]


@dataclass
class RepairHit:
    record: RepairRecord
    similarity: float
    prior: float
    score: float

    def as_prompt(self) -> str:
        """Compact few-shot block: the planner sees the fix, not the whole transcript."""
        verdict = "GREEN" if self.record.success else "STILL RED"
        tail = self.record.diff.strip().splitlines()[-12:]
        return (
            f"[{self.score:.3f}] class={self.record.failure_class} attempts={self.record.attempts} {verdict}\n"
            + "\n".join(tail)
        )


class RepairMemory:
    def __init__(
        self,
        path: Path | str = ":memory:",
        *,
        embedder=None,
        backend: str = "hashing",
        dim: int = 96,
        ivf_threshold: int = IVF_THRESHOLD,
    ) -> None:
        if np is None:
            raise ImportError(f"repair memory needs the ml extra ({_ML_IMPORT_ERROR})")
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.embedder = embedder or build_embedder(backend, dim=dim)
        self.backend_name = getattr(self.embedder, "name", backend)
        self.ivf_threshold = ivf_threshold
        self._records: dict[str, RepairRecord] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._index_dirty = True
        self._index = None

    # ---------------------------------------------------------------- write side
    def _row_to_record(self, row: sqlite3.Row) -> RepairRecord:
        return RepairRecord(
            traceback=row["traceback"],
            diff=row["diff"],
            failure_class=row["failure_class"],
            success=bool(row["success"]),
            attempts=int(row["attempts"]),
            ts=float(row["ts"]),
            id=row["id"],
            meta=json.loads(row["meta"] or "{}"),
        )

    def add(self, record: RepairRecord) -> RepairRecord:
        """Insert, or fold a repeat attempt into the existing row and update its counters."""
        existing = self.conn.execute("SELECT * FROM repairs WHERE id=?", (record.id,)).fetchone()
        if existing:
            self.conn.execute(
                "UPDATE repairs SET success=?, attempts=?, ts=?, meta=? WHERE id=?",
                (
                    int(bool(record.success)),
                    int(existing["attempts"]) + int(record.attempts),
                    record.ts,
                    json.dumps({**json.loads(existing["meta"] or "{}"), **record.meta}),
                    record.id,
                ),
            )
            self.conn.commit()
            self._index_dirty = True
            return record
        vector = np.asarray(self.embedder.embed_one(record.traceback), dtype=np.float32)
        self.conn.execute(
            "INSERT INTO repairs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                record.id,
                record.ts,
                record.failure_class,
                record.traceback,
                record.diff,
                int(bool(record.success)),
                int(record.attempts),
                record.tb_hash,
                record.diff_hash,
                self.backend_name,
                vector.tobytes(),
                json.dumps(record.meta),
            ),
        )
        self.conn.commit()
        self._index_dirty = True
        return record

    def record(self, traceback: str, diff: str, *, failure_class: str, success: bool, attempts: int = 1, **meta) -> RepairRecord:
        return self.add(RepairRecord(traceback, diff, failure_class, success, attempts, meta=dict(meta)))

    def bump(self, failure_class: str, *, success: bool) -> None:
        key = failure_class or "unknown"
        self.conn.execute(
            "INSERT INTO counters(key, hits, wins) VALUES(?,1,?) "
            "ON CONFLICT(key) DO UPDATE SET hits=hits+1, wins=wins+?",
            (key, int(success), int(success)),
        )
        self.conn.commit()

    # ---------------------------------------------------------------- read side
    def _load(self) -> None:
        if self._records and not self._index_dirty:
            return
        expected = int(getattr(self.embedder, "dim", 0) or 0)
        self._records, self._vectors = {}, {}
        drifted: list[tuple[str, np.ndarray]] = []
        for row in self.conn.execute("SELECT * FROM repairs ORDER BY ts"):
            rec = self._row_to_record(row)
            self._records[rec.id] = rec
            vector = np.frombuffer(row["embedding"], dtype=np.float32)
            if expected and (vector.shape[0] != expected or row["embedder"] != self.backend_name):
                vector = np.asarray(self.embedder.embed_one(row["traceback"]), dtype=np.float32)
                drifted.append((rec.id, vector))
            self._vectors[rec.id] = vector
        if drifted:
            for doc_id, vector in drifted:
                self.conn.execute(
                    "UPDATE repairs SET embedding=?, embedder=? WHERE id=?",
                    (vector.tobytes(), self.backend_name, doc_id),
                )
            self.conn.commit()
            log.warning("re-embedded rows written by another backend", extra={"rows": len(drifted)})
        self._index_dirty = False
        self._index = None

    def _build_index(self):
        self._load()
        if not self._records:
            return None
        ids = list(self._records)
        vectors = np.vstack([self._vectors[i] for i in ids])
        if len(ids) >= self.ivf_threshold:
            index = IvfIndex(ids, vectors, n_lists=0, n_probe=4)
            index.build()
            return index
        return FlatIndex(ids, vectors)

    @property
    def index(self):
        if self._index is None or self._index_dirty:
            self._index = self._build_index()
        return self._index

    def _prior(self, record: RepairRecord) -> float:
        """Laplace-smoothed P(this diff leads to green | its failure class)."""
        row = self.conn.execute(
            "SELECT hits, wins FROM counters WHERE key=?", (record.failure_class or "unknown",)
        ).fetchone()
        if not row:
            stat = self.conn.execute(
                "SELECT COUNT(*) n, SUM(success) s FROM repairs WHERE failure_class=?",
                (record.failure_class or "unknown",),
            ).fetchone()
            n, s = int(stat["n"] or 0), int(stat["s"] or 0)
        else:
            n, s = int(row["hits"]), int(row["wins"])
        return (s + 1) / (n + 2)

    def recall(
        self,
        query: str,
        k: int = 3,
        *,
        failure_class: str | None = None,
        min_similarity: float = 0.08,
        diversify: bool = True,
        weights: dict[str, float] | None = None,
    ) -> list[RepairHit]:
        index = self.index
        if index is None:
            return []
        w = {"similarity": 0.75, "prior": 0.25, **(weights or {})}
        vector = np.asarray(self.embedder.embed_one(query), dtype=np.float32)
        pool = index.search(vector, max(k * 6, 12))
        hits: list[RepairHit] = []
        for doc_id, similarity in pool:
            record = self._records[doc_id]
            if similarity < min_similarity:
                continue
            if failure_class and record.failure_class != failure_class:
                continue
            prior = self._prior(record)
            score = w["similarity"] * similarity + w["prior"] * prior
            hits.append(RepairHit(record, float(similarity), float(prior), float(score)))
        if diversify and len(hits) > k:
            chosen = mmr(vector, [(h.record.id, self._vectors[h.record.id]) for h in hits], k)
            picked = {cid for cid, _ in chosen}
            hits = [h for h in hits if h.record.id in picked] + [h for h in hits if h.record.id not in picked]
        return hits[:k]

    def prompt_block(self, query: str, k: int = 3, **kw) -> str:
        hits = self.recall(query, k, **kw)
        if not hits:
            return ""
        body = "\n\n".join(h.as_prompt() for h in hits)
        return f"Similar past repairs (best first):\n{body}"

    def context(self, query: str, *, k: int = 3, failure_class: str | None = None) -> str:
        """Recall for the prompt, degrading to no hints instead of an exception.

        The store is an accelerator: a locked database, a half-written index or a
        missing optional dependency has to leave the planner with a plain
        traceback, which is exactly what it had before memory existed.
        """
        if not (query or "").strip() or k <= 0:
            return ""
        try:
            return self.prompt_block(query, k, failure_class=failure_class)
        except Exception as exc:  # pragma: no cover - depends on the host's sqlite/fs
            log.warning("recall skipped", extra={"error": str(exc)})
            return ""

    # ------------------------------------------------------------- introspection
    def stats(self) -> dict:
        rows = self.conn.execute("SELECT COUNT(*) n, SUM(success) s, AVG(attempts) a FROM repairs").fetchone()
        n = int(rows["n"] or 0)
        s = int(rows["s"] or 0)
        classes = self.conn.execute(
            "SELECT failure_class k, COUNT(*) n, SUM(success) s FROM repairs GROUP BY failure_class ORDER BY n DESC"
        ).fetchall()
        index = self.index
        return {
            "records": n,
            "success_rate": round(s / n, 4) if n else 0.0,
            "avg_attempts": round(float(rows["a"] or 0), 3),
            "embedder": self.backend_name,
            "dim": int(getattr(self.embedder, "dim", 0)),
            "index": type(index).__name__ if index is not None else "empty",
            "by_class": {r["k"]: {"n": int(r["n"]), "success": int(r["s"] or 0)} for r in classes},
        }

    def export_jsonl(self) -> str:
        self._load()
        return "\n".join(json.dumps(asdict(r), ensure_ascii=False) for r in self._records.values())

    def fit_embedder(self, corpus: Iterable[str], *, model_path: Path | str | None = None) -> str:
        """Refit the learned backend on accumulated tracebacks and reindex everything."""
        texts = [t for t in corpus if t and t.strip()]
        if len(texts) < 8:
            return "skipped: need at least 8 tracebacks to fit"
        from .embeddings import TfidfLsaEmbedder

        embedder = TfidfLsaEmbedder(dim=self.embedder.dim if getattr(self.embedder, "name", "") != "hashing" else 96)
        embedder.fit(texts)
        if model_path:
            embedder.save(model_path)
        self.embedder = embedder
        self.backend_name = embedder.name
        rows = list(self.conn.execute("SELECT id, traceback FROM repairs"))
        for row in rows:
            vector = np.asarray(embedder.embed_one(row["traceback"]), dtype=np.float32)
            self.conn.execute(
                "UPDATE repairs SET embedding=?, embedder=? WHERE id=?", (vector.tobytes(), embedder.name, row["id"])
            )
        self.conn.commit()
        self._index_dirty = True
        return f"refit {embedder.name} on {len(texts)} docs, dim={embedder.dim}"

    def close(self) -> None:
        self.conn.close()

    def __len__(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) n FROM repairs").fetchone()["n"])

    def __enter__(self) -> "RepairMemory":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def seed_memory(memory: RepairMemory, records: Sequence[RepairRecord] | None = None) -> int:
    """Load the shipped repair examples so demos and evals have a non-empty store."""
    from .corpus import REPAIR_EXAMPLES

    count = 0
    for item in records or REPAIR_EXAMPLES:
        memory.record(
            str(item["traceback"]),
            str(item["diff"]),
            failure_class=str(item["failure_class"]),
            success=bool(item["success"]),
            attempts=int(item["attempts"]),
            seeded=True,
        )
        memory.bump(str(item["failure_class"]), success=bool(item["success"]))
        count += 1
    return count


def observe(
    memory: RepairMemory | None,
    traceback: str,
    diff: str,
    *,
    failure_class: str,
    success: bool,
    attempts: int = 1,
    **meta,
) -> None:
    """Store one verdict. Shared by the three loops so none of them re-implements it.

    Both the write and the counter bump are swallowed on purpose: learning that a
    repair failed is worth having, but not at the price of the run that is already
    green and about to return.
    """
    if memory is None or not (traceback or "").strip() or not (diff or "").strip():
        return
    try:
        memory.record(
            traceback, diff, failure_class=failure_class or "unknown", success=success,
            attempts=attempts, **meta,
        )
        memory.bump(failure_class or "unknown", success=success)
    except Exception as exc:  # pragma: no cover - depends on the host's sqlite/fs
        log.warning("memory write skipped", extra={"error": str(exc)})


def open_memory(settings, *, seed: bool = True) -> RepairMemory | None:
    """Build the episodic store from Settings, or answer None and change nothing.

    Memory is opt-in because it pulls in numpy. An empty store is seeded with the
    shipped repair examples, so the first demo run already has something to recall.
    """
    if not getattr(settings, "use_ml", False):
        return None
    try:
        memory = RepairMemory(getattr(settings, "memory_path", ":memory:"))
    except Exception as exc:  # numpy/scipy missing, or the path is not writable
        log.warning("memory disabled", extra={"error": str(exc), "path": str(getattr(settings, "memory_path", ""))})
        return None
    if seed and len(memory) == 0:
        try:
            seed_memory(memory)
        except Exception as exc:  # pragma: no cover - corpus is data, not logic
            log.warning("memory seeding skipped", extra={"error": str(exc)})
    return memory
