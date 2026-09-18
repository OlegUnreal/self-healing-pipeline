"""Embedding backends must stay deterministic and offline-friendly.

The whole ML stack rests on one property: `embed()` returns the same float32 row
for the same text under the same fitted state, without touching the network.
"""
from __future__ import annotations

import numpy as np
import pytest

from self_healing.embeddings import (
    BACKENDS,
    EmbeddingNotFitted,
    HashingEmbedder,
    OpenAIEmbedder,
    TfidfLsaEmbedder,
    build_embedder,
    l2_normalise,
    tokenize,
)


def test_tokenize_keeps_underscores_and_drops_stopwords():
    tokens = tokenize("TimeoutError: retry_budget_exceeded failed after 3 line")
    assert "retry_budget_exceeded" in tokens
    assert "timeouterror" in tokens
    assert "line" not in tokens
    assert not any(t.isdigit() for t in tokens)


def test_tokenize_is_lowercase_and_idempotent():
    text = "NameError: name 'Cfg' is not defined"
    assert tokenize(text) == tokenize(text)
    assert all(t == t.lower() for t in tokenize(text))


def test_l2_normalise_gives_unit_rows_and_survives_zeros():
    out = l2_normalise(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    assert np.isclose(float(np.linalg.norm(out[0])), 1.0)
    assert np.allclose(out[1], 0.0)


def test_hashing_embedder_is_deterministic_and_unit_norm():
    emb = HashingEmbedder(dim=64).fit()
    a = emb.embed(["assertion failed: expected 1 got 0"])
    b = emb.embed(["assertion failed: expected 1 got 0"])
    assert a.dtype == np.float32
    assert np.allclose(a, b)
    assert np.isclose(float(np.linalg.norm(a[0])), 1.0)
    assert emb.embed_one("x").shape == (64,)


def test_tfidf_lsa_requires_fit():
    with pytest.raises(EmbeddingNotFitted):
        TfidfLsaEmbedder(dim=8).embed(["anything"])


def test_tfidf_lsa_reduces_dimension_and_keeps_semantic_order():
    corpus = [
        "TimeoutError: request to api timed out after 30s",
        "ConnectionError: socket reset while calling api",
        "IndentationError: unexpected indent in parser",
        "ZeroDivisionError: division by zero in stats",
    ]
    emb = TfidfLsaEmbedder(dim=3).fit(corpus)
    X = emb.embed(corpus)
    assert X.shape == (4, 3)
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-5)
    sims = X @ X.T
    assert sims[0, 1] > sims[0, 2]


def test_tfidf_lsa_save_load_roundtrip(tmp_path):
    corpus = ["name error: cfg missing", "type error: str is not int", "timeout on api"]
    emb = TfidfLsaEmbedder(dim=4).fit(corpus)
    path = emb.save(tmp_path / "embed.joblib")
    restored = TfidfLsaEmbedder.load(path)
    assert np.allclose(emb.embed(corpus), restored.embed(corpus), atol=1e-6)


@pytest.mark.parametrize("backend", ["hashing", "tfidf-lsa"])
def test_build_embedder_returns_named_ready_backends(backend):
    emb = build_embedder(backend, corpus=["a error", "b error", "c error"], dim=32)
    assert emb.name == backend
    assert emb.embed_one("a error").shape[0] > 0


def test_build_embedder_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown embedding backend"):
        build_embedder("faiss")


def test_openai_backend_exists_but_is_never_default():
    assert "openai" in BACKENDS
    assert build_embedder("hashing").name != OpenAIEmbedder().name


def test_fit_requires_a_minimum_corpus_and_caps_the_dimension():
    with pytest.raises(ValueError, match="at least 3 documents"):
        TfidfLsaEmbedder(dim=64).fit(["only one document here"])
    emb = TfidfLsaEmbedder(dim=64).fit(["a error here", "b error here", "c error here"])
    assert 0 < emb.embed(["a error here"]).shape[1] <= 64


def test_unigram_control_blinds_the_embedder_to_token_order():
    """The classifier's bag-of-words ablation relies on this: word bigrams already
    encode which exception came first, so a control that keeps them isolates nothing.

    One caveat the assertions make explicit: a bigram only carries signal when that
    exact ordering was present while fitting. Reorder a traceback into a sequence the
    corpus never contained and the production embedder goes blind too, which is why
    the ablation is measured on the training split instead of on invented pairs.
    """
    first = "ValueError: bad payload\nTypeError: cannot parse"
    second = "TypeError: cannot parse\nValueError: bad payload"
    filler = [
        "KeyError: 'user_id'\nAttributeError: 'NoneType' object has no attribute 'get'",
        "TypeError: cannot parse\nIndexError: list index out of range",
        "ConnectionError: reset by peer\nTimeoutError: deadline exceeded",
        "ZeroDivisionError: division by zero\nValueError: bad payload",
        "AttributeError: 'NoneType' object has no attribute 'get'\nKeyError: 'user_id'",
        "IndexError: list index out of range\nTypeError: cannot parse",
    ]
    corpus = [first, second, *filler]

    control = TfidfLsaEmbedder(dim=16, word_ngram_range=(1, 1), char_ngram_range=None).fit(corpus)
    ordered = TfidfLsaEmbedder(dim=16).fit(corpus)
    assert not any(" " in f for f in control._vectorizer.word.vocabulary_)
    assert any(" " in f for f in ordered._vectorizer.word.vocabulary_)

    a, b = control.embed([first, second])
    assert float(a @ b) == pytest.approx(1.0, abs=1e-5), "a bag of words cannot order anything"
    x, y = ordered.embed([first, second])
    assert float(x @ y) < 0.999, "production features must separate the two orderings"
