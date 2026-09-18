"""Classifier quality gates.

These tests assert measured numbers, not just "it runs". The interesting claim is
narrow and deliberate: on single-exception tracebacks the keyword ladder is already
optimal (both sides score 1.0), and the model only wins on *chained* tracebacks where
the label is the outermost exception - there it reaches macro-F1 0.73 against 0.50 for
the ladder and 0.61 for a genuine unigram bag of words. Gate that gap instead of
hiding it.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from self_healing.classifier import (
    RULE_ORDER,
    TracebackClassifier,
    build_classifier,
    classification_report,
    expected_calibration_error,
    model_card,
    rule_class,
    structured_view,
    view,
    _stratified_split,
)
from self_healing.corpus import ALL_TRACEBACKS, LABELS, SEED_TRACEBACKS, is_hard
from self_healing.embeddings import HashingEmbedder, build_embedder

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def trained() -> dict:
    """One deterministic train/eval pass for the whole module (~2s)."""
    classifier = TracebackClassifier(embedder=build_embedder("tfidf-lsa", corpus=[t for t, _ in ALL_TRACEBACKS], dim=96))
    report = classifier.train_from_corpus(ALL_TRACEBACKS, test_size=0.3, seed=7)
    classifier._report = report  # type: ignore[attr-defined]
    return report


# --------------------------------------------------------------------- the gates
def test_model_beats_the_rule_ladder_only_where_ordering_matters(trained):
    assert trained["chained_model_macro_f1"] >= trained["chained_rule_macro_f1"] + 0.15, (
        "the ordering features no longer beat the keyword ladder: "
        f"model={trained['chained_model_macro_f1']} rules={trained['chained_rule_macro_f1']}"
    )


def test_overall_quality_floor(trained):
    assert trained["macro_f1"] >= 0.80, trained["macro_f1"]
    assert trained["accuracy"] >= 0.80, trained["accuracy"]


def test_chained_segment_absolute_floor(trained):
    assert trained["chained_model_macro_f1"] >= 0.65, trained["chained_model_macro_f1"]
    assert trained["chained_n"] >= 25


def test_single_exception_segment_is_an_honest_tie(trained):
    assert trained["single_exception_model_macro_f1"] >= 0.95
    assert trained["single_exception_rule_macro_f1"] >= 0.95


def test_probabilities_are_useful_for_a_thresholding_planner(trained):
    assert trained["ece"] <= 0.25, trained["ece"]
    assert trained["calibration"] == "sigmoid-cv3", trained["calibration"]


def test_exception_order_features_are_the_reason_it_works(trained):
    assert trained["chained_model_macro_f1"] >= trained["bag_only_chained_macro_f1"] + 0.05, (
        "a unigram bag of words matched the structured view; the feature engineering is dead weight: "
        f"structured={trained['chained_model_macro_f1']} bag={trained['bag_only_chained_macro_f1']}"
    )
    assert trained["features"] == "tfidf-lsa+exception-order"


def test_report_is_reproducible_for_a_fixed_seed():
    first = TracebackClassifier(embedder=build_embedder("tfidf-lsa", corpus=[t for t, _ in ALL_TRACEBACKS], dim=96))
    second = TracebackClassifier(embedder=build_embedder("tfidf-lsa", corpus=[t for t, _ in ALL_TRACEBACKS], dim=96))
    a = first.train_from_corpus(ALL_TRACEBACKS, test_size=0.3, seed=7)
    b = second.train_from_corpus(ALL_TRACEBACKS, test_size=0.3, seed=7)
    for key in ("accuracy", "macro_f1", "chained_model_macro_f1", "ece"):
        assert a[key] == b[key], key


_SPLIT_PROBE = (
    "import hashlib;"
    "from self_healing.classifier import _stratified_split;"
    "from self_healing.corpus import ALL_TRACEBACKS;"
    "_,test=_stratified_split(ALL_TRACEBACKS,test_size=0.3,seed=7);"
    "print(hashlib.sha1('|'.join(t for t,_ in test).encode('utf-8')).hexdigest())"
)


def test_the_split_does_not_depend_on_the_hash_seed():
    """The splitter once iterated a set of labels, so `PYTHONHASHSEED` decided the shuffle
    draws, the split, and every published metric (accuracy ranged 0.75-0.86 between runs).
    In-process determinism cannot see it, so pay for two interpreters."""
    digests = set()
    for seed in ("0", "7"):
        done = subprocess.run(
            [sys.executable, "-c", _SPLIT_PROBE],
            cwd=str(REPO_ROOT),
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert done.returncode == 0, done.stderr[-400:]
        digests.add(done.stdout.strip())
    assert len(digests) == 1, f"hold-out membership moved with the hash seed: {digests}"


# ------------------------------------------------------------------ feature view
def test_structured_view_encodes_order_not_just_membership():
    inner_first = "ValueError: bad payload\nTypeError: cannot parse"
    outer_first = "TypeError: cannot parse\nValueError: bad payload"
    a, b = structured_view(inner_first), structured_view(outer_first)
    assert "lastexc_typeerror" in a
    assert "lastexc_valueerror" in b
    assert "pair_valueerror_typeerror" in a
    assert "pair_typeerror_valueerror" in b
    assert a != b


def test_structured_view_handles_no_exception_and_single_exception():
    assert structured_view("nothing useful here") == "exc_none"
    single = structured_view("NameError: undefined cfg")
    assert "onlyexc_nameerror" in single
    assert "nexc_1" in single


def test_view_appends_the_structured_tail_only_when_asked():
    text = "NameError: cfg"
    assert view(text, structured=False) == text
    assert view(text).endswith(structured_view(text))


def test_tokenizer_keeps_structured_tokens_whole():
    from self_healing.embeddings import tokenize

    tokens = set(tokenize(view("ValueError: a\nTypeError: b")))
    assert "pair_valueerror_typeerror" in tokens
    assert "lastexc_typeerror" in tokens


# ------------------------------------------------------------------- rule parity
@pytest.mark.parametrize(
    "text,expected",
    [
        ("SyntaxError: invalid syntax", "syntax_error"),
        ("IndentationError: unexpected indent", "indentation_error"),
        ("ModuleNotFoundError: No module named 'x'", "import_error"),
        ("NameError: name 'cfg' is not defined", "name_error"),
        ("AssertionError: assert 1 == 2", "assertion_failure"),
        ("TypeError: unsupported operand", "type_error"),
        ("TimeoutError: read timed out", "timeout"),
    ],
)
def test_rule_class_matches_the_shipped_ladder(text, expected):
    assert rule_class(text) == expected


def test_rule_class_falls_back_on_exit_code():
    assert rule_class("", exit_code=0) == "ok"
    assert rule_class("", exit_code=1) == "runtime_error"
    assert rule_class("anything", timed_out=True) == "timeout"
    assert {label for _, label in RULE_ORDER} <= set(LABELS)


# ---------------------------------------------------------------------- plumbing
def test_stratified_split_keeps_every_class_on_both_sides():
    train, test = _stratified_split(ALL_TRACEBACKS, test_size=0.3, seed=7)
    assert len(train) + len(test) == len(ALL_TRACEBACKS)
    assert set(y for _, y in train) == set(LABELS) == set(y for _, y in test)
    assert abs(len(test) / len(ALL_TRACEBACKS) - 0.3) < 0.06


def test_classification_report_math():
    report = classification_report(["a", "a", "b"], ["a", "b", "b"])
    assert report["accuracy"] == pytest.approx(2 / 3, abs=1e-3)
    assert report["per_class"]["a"]["f1"] == pytest.approx(0.6667, abs=1e-3)
    assert report["macro_f1"] == pytest.approx((report["per_class"]["a"]["f1"] + report["per_class"]["b"]["f1"]) / 2)


def test_expected_calibration_error_bounds():
    perfect = np.eye(3)
    assert expected_calibration_error(perfect, np.array([0, 1, 2])) == pytest.approx(0.0)
    overconfident = np.array([[0.99, 0.005, 0.005]] * 4)
    assert expected_calibration_error(overconfident, np.zeros(4, dtype=int) + 1) > 0.7


def test_centroid_fallback_predicts_without_sklearn_model():
    classifier = TracebackClassifier(embedder=HashingEmbedder(dim=64), backend="centroid")
    classifier.fit(SEED_TRACEBACKS)
    assert classifier.predict("NameError: name 'cfg' is not defined") in LABELS
    probs = classifier.predict_proba(["TypeError: unsupported operand"])
    assert np.isclose(float(probs.sum()), 1.0)


def test_unfitted_classifier_refuses_to_predict():
    with pytest.raises(RuntimeError, match="not fitted"):
        TracebackClassifier(embedder=HashingEmbedder(dim=16), backend="centroid").predict("boom")


def test_save_load_roundtrip_preserves_predictions_and_metrics(trained, tmp_path):
    classifier = TracebackClassifier(embedder=build_embedder("tfidf-lsa", corpus=[t for t, _ in ALL_TRACEBACKS], dim=96))
    classifier.train_from_corpus(ALL_TRACEBACKS, test_size=0.3, seed=7)
    texts = [t for t, _ in ALL_TRACEBACKS[::13]]
    path = classifier.save(tmp_path / "model" / "clf.joblib")
    restored = TracebackClassifier.load(path)
    assert restored.predict_one(texts) == classifier.predict_one(texts)
    assert restored.metrics["macro_f1"] == trained["macro_f1"]
    assert restored.structured is True


def test_explain_returns_sorted_top_labels():
    classifier = TracebackClassifier(embedder=build_embedder("tfidf-lsa", corpus=[t for t, _ in ALL_TRACEBACKS], dim=64))
    classifier.fit(ALL_TRACEBACKS)
    top = classifier.explain("AssertionError: assert 0 == 1", top=3)
    assert len(top) == 3
    assert top[0][1] >= top[1][1] >= top[2][1]
    assert top[0][0] in LABELS


def test_build_classifier_caches_and_reuses_the_artifact(tmp_path):
    path = tmp_path / "clf.joblib"
    fresh = build_classifier(model_path=path, backend="hashing", dim=48)
    assert path.exists()
    assert fresh.metrics.get("macro_f1", 0) > 0.5
    assert is_hard(ALL_TRACEBACKS[100][0])
    loaded = build_classifier(model_path=path, auto_train=False)
    assert loaded is not None
    assert loaded.predict("TypeError: bad operand") == fresh.predict("TypeError: bad operand")


def test_build_classifier_can_opt_out_of_training():
    assert build_classifier(model_path="nope/missing.joblib", auto_train=False) is None


def test_model_card_states_the_comparison(trained):
    classifier = TracebackClassifier(embedder=build_embedder("tfidf-lsa", corpus=[t for t, _ in ALL_TRACEBACKS], dim=96))
    classifier.metrics = trained
    card = model_card(classifier)
    assert "rule baseline" in card
    assert "chained tracebacks" in card
    assert "| class | precision | recall | f1 | support |" in card
    assert str(trained["macro_f1"]) in card
