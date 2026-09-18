"""Labelled corpus integrity, including the ordering-dependent subset.

`HARD_TRACEBACKS` exists to make the classifier earn its keep: the label is the
*outermost* exception of a chained traceback, so the same two exception names in a
different order mean different classes. A keyword ladder cannot represent that.
"""
from __future__ import annotations

import re

import pytest

from self_healing.classifier import rule_class
from self_healing.corpus import (
    ALL_TRACEBACKS,
    HARD_TRACEBACKS,
    LABELS,
    REPAIR_EXAMPLES,
    SEED_TRACEBACKS,
    hard_share,
    is_hard,
    seed_of,
)

EXC_RE = re.compile(r"\b([A-Z][A-Za-z]*(?:Error|Exception|Failure|Timeout))\b")


def test_corpus_sizes_and_membership():
    assert len(SEED_TRACEBACKS) >= 60
    assert len(HARD_TRACEBACKS) > 90
    assert ALL_TRACEBACKS == SEED_TRACEBACKS + HARD_TRACEBACKS
    assert all(label in LABELS for _, label in ALL_TRACEBACKS)


def test_no_duplicate_texts_and_no_empty_entries():
    texts = [t for t, _ in ALL_TRACEBACKS]
    assert len(set(texts)) == len(texts), "duplicate tracebacks leak across the train/test split"
    assert all(t.strip() for t in texts)


def test_every_label_is_learnable_and_represented_in_hard_subset():
    per_class = {l: sum(1 for _, y in ALL_TRACEBACKS if y == l) for l in LABELS}
    assert min(per_class.values()) >= 10, per_class
    hard_per_class = {l: sum(1 for _, y in HARD_TRACEBACKS if y == l) for l in LABELS}
    assert min(hard_per_class.values()) >= 8, hard_per_class


def test_hard_subset_is_majority_and_seeds_are_minority():
    assert hard_share(HARD_TRACEBACKS) == 1.0
    assert hard_share(ALL_TRACEBACKS) > 0.5
    assert hard_share(SEED_TRACEBACKS) < 0.25


def test_chained_examples_are_genuinely_chained_and_ordered():
    for text, label in HARD_TRACEBACKS:
        names = EXC_RE.findall(text)
        assert len(set(names)) > 1 or is_hard(text)
        assert any("above exception" in line for line in text.splitlines()), "missing chain marker"
    sample = HARD_TRACEBACKS[0][0]
    assert "The above exception was the direct cause" in sample or "During handling" in sample


def test_label_matches_the_outermost_exception():
    for text, label in HARD_TRACEBACKS:
        matches = list(EXC_RE.finditer(text))
        outer_line = text[matches[-1].start() :].splitlines()[0]
        assert rule_class(outer_line) == label, f"{outer_line!r} read as {rule_class(outer_line)} != {label}"


def test_first_match_ladder_cannot_read_the_chained_subset():
    correct = sum(1 for text, label in HARD_TRACEBACKS if rule_class(text) == label)
    accuracy = correct / len(HARD_TRACEBACKS)
    assert accuracy < 0.6, f"the chained subset stopped being ordering-dependent: rules score {accuracy}"


def test_is_hard_flags_multi_exception_traces_only():
    assert not is_hard("ValueError: single failure\n")
    assert is_hard(
        "ValueError: a\nThe above exception was the direct cause of the following exception:\nTypeError: b\n"
    )


def test_seed_of_is_deterministic_per_label():
    assert seed_of("timeout", 0) == seed_of("timeout", 0)
    assert seed_of("name_error", 0) in [t for t, _ in SEED_TRACEBACKS]


def test_repair_examples_match_classifier_labels():
    assert len(REPAIR_EXAMPLES) >= 6
    for example in REPAIR_EXAMPLES:
        assert example["failure_class"] in LABELS
        assert {"traceback", "diff", "success", "attempts"} <= set(example)


@pytest.mark.parametrize("label", LABELS)
def test_each_label_has_chained_coverage(label):
    assert any(y == label for _, y in HARD_TRACEBACKS)
