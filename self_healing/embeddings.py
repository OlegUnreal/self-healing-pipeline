"""Text embedders for repair memory and failure classification.

One contract, three backends: `fit(corpus)` then `embed(texts) -> float32 rows, L2
normalised`. `hashing` needs only numpy so tests never need a trained model on disk;
`tfidf-lsa` is a fitted TruncatedSVD projection and is what the shipped models use;
`openai` is opt-in for live runs.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

log = logging.getLogger("self_healing.embeddings")

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z_0-9]*|\d+|[^\w\s]")
_STOP = frozenset(
    """a an the of to in is are was were and or for on by as at it this that with from
    def return assert import raise error line file""".split()
)

BACKENDS = ("hashing", "tfidf-lsa", "openai")


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in _STOP or tok.isdigit():
            continue
        out.append(tok)
        stem = tok.rstrip("0123456789_")
        if stem != tok and stem:
            out.append(stem)
    return out


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.maximum(norms, 1e-12)).astype(np.float32)


_normalise = l2_normalise


class EmbeddingNotFitted(RuntimeError):
    pass


@dataclass
class HashingEmbedder:
    """Deterministic signed char/word hashing. No training data required."""

    dim: int = 256
    name: str = "hashing"
    version: int = 1
    _out: np.ndarray = field(default=None, repr=False, init=False)  # type: ignore[assignment]

    def fit(self, corpus: Iterable[str] | None = None) -> "HashingEmbedder":
        return self

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            counts: dict[int, float] = {}
            for tok in tokenize(text):
                for feat in (tok, f"#{tok}"):
                    h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
                    idx = h % self.dim
                    sign = 1.0 if (h >> 63) & 1 == 0 else -1.0
                    counts[idx] = counts.get(idx, 0.0) + sign
            rows[i, list(counts)] = list(counts.values())
        self._out = _normalise(rows)
        return self._out

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def _identity(text: str) -> str:
    """Module-level on purpose: a lambda here would make the fitted vectorizer unpicklable."""
    return text


class _TfidfUnion:
    """Word + char-ngram TF-IDF concatenated into one sparse matrix.

    `char` may be None, which leaves a word-only matrix - that is how the bag-of-words
    ablation drops every order-sensitive feature in one step.
    """

    def __init__(self, word, char) -> None:
        self.word = word
        self.char = char

    def fit_transform(self, texts: list[str]):
        from scipy.sparse import hstack

        self.word.fit(texts)
        if self.char is None:
            return self.word.transform(texts)
        self.char.fit(texts)
        return hstack([self.word.transform(texts), self.char.transform(texts)])

    def transform(self, texts: list[str]):
        from scipy.sparse import hstack

        if self.char is None:
            return self.word.transform(texts)
        return hstack([self.word.transform(texts), self.char.transform(texts)])


@dataclass
class TfidfLsaEmbedder:
    """Word+char TF-IDF projected by TruncatedSVD (probabilistic LSA)."""

    dim: int = 96
    name: str = "tfidf-lsa"
    version: int = 1
    seed: int = 7
    char_features: int = 40_000
    word_features: int = 20_000
    word_ngram_range: tuple[int, int] = (1, 2)
    char_ngram_range: tuple[int, int] | None = (3, 5)
    _vectorizer: Any = field(default=None, repr=False, init=False)
    _svd: Any = field(default=None, repr=False, init=False)
    _fitted: bool = field(default=False, repr=False, init=False)

    @property
    def is_fitted(self) -> bool:
        return bool(self._fitted)

    def _build(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        try:
            word = TfidfVectorizer(
                tokenizer=tokenize,
                preprocessor=_identity,
                token_pattern=None,
                analyzer="word",
                ngram_range=self.word_ngram_range,
                sublinear_tf=True,
                min_df=1,
                max_features=self.word_features,
            )
        except TypeError:  # sklearn < 1.4 rejects token_pattern=None
            word = TfidfVectorizer(
                ngram_range=self.word_ngram_range,
                sublinear_tf=True,
                min_df=1,
                max_features=self.word_features,
            )
        if not self.char_ngram_range:
            return word, None
        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.char_ngram_range,
            sublinear_tf=True,
            min_df=1,
            max_features=self.char_features,
        )
        return word, char

    def fit(self, corpus: Iterable[str]) -> "TfidfLsaEmbedder":
        from sklearn.decomposition import TruncatedSVD

        texts = [c for c in corpus if c and c.strip()]
        if len(texts) < 3:
            raise ValueError("tfidf-lsa needs at least 3 documents to fit")
        word, char = self._build()
        union = _TfidfUnion(word, char)
        matrix = union.fit_transform(texts)
        n_components = int(min(self.dim, max(2, min(matrix.shape) - 1)))
        svd = TruncatedSVD(n_components=n_components, random_state=self.seed)
        svd.fit(matrix)
        self._vectorizer = union
        self._svd = svd
        self.dim = int(n_components)
        self._fitted = True
        log.info(
            "fitted tfidf-lsa embedder",
            extra={
                "docs": len(texts),
                "dim": self.dim,
                "explained_variance": round(float(svd.explained_variance_ratio_.sum()), 4),
            },
        )
        return self

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise EmbeddingNotFitted("TfidfLsaEmbedder.fit() has not been called")
        matrix = self._vectorizer.transform(list(texts))
        return _normalise(self._svd.transform(matrix).astype(np.float32))

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def save(self, path: Path | str) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self._vectorizer, "svd": self._svd, "dim": self.dim, "version": self.version}, path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "TfidfLsaEmbedder":
        import joblib

        blob = joblib.load(Path(path))
        obj = cls(dim=int(blob.get("dim", cls.dim)))
        obj._vectorizer = blob["vectorizer"]
        obj._svd = blob["svd"]
        obj.version = int(blob.get("version", 1))
        obj._fitted = True
        return obj


@dataclass
class OpenAIEmbedder:
    """Live embeddings behind an env flag; batched and retried by the SDK."""

    model: str = "text-embedding-3-small"
    dim: int = 1536
    name: str = "openai"
    version: int = 1
    api_key: str = ""

    def fit(self, corpus: Iterable[str] | None = None) -> "OpenAIEmbedder":
        return self

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key or None)
        out: list[list[float]] = []
        for start in range(0, len(texts), 128):
            batch = list(texts[start : start + 128])
            res = client.embeddings.create(model=self.model, input=batch)
            out.extend(item.embedding for item in res.data)
        return _normalise(np.asarray(out, dtype=np.float32))

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def build_embedder(
    backend: str = "hashing",
    *,
    corpus: Iterable[str] | None = None,
    dim: int = 96,
    model_path: Path | str | None = None,
    api_key: str = "",
):
    """Return a ready embedder, loading or fitting a persisted model when needed."""
    path = Path(model_path) if model_path else None
    if backend == "hashing":
        return HashingEmbedder(dim=max(16, dim * 2))
    if backend == "openai":
        return OpenAIEmbedder(dim=dim, api_key=api_key)
    if backend != "tfidf-lsa":
        raise ValueError(f"unknown embedding backend {backend!r}; expected one of {BACKENDS}")
    if path is not None and path.exists():
        try:
            return TfidfLsaEmbedder.load(path)
        except Exception as exc:  # corrupt artifact must not take the pipeline down
            log.warning("embedder artifact unreadable, refitting", extra={"error": str(exc)})
    embedder = TfidfLsaEmbedder(dim=dim)
    embedder.fit(list(corpus or []))
    if path is not None:
        embedder.save(path)
    return embedder
