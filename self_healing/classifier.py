"""Learned failure classifier: root cause is the last frame, not the first keyword.

`verifier.classify_output` is a first-match keyword ladder, so a transcript that raises
`ModuleNotFoundError` in an optional import and then dies on `NameError` is reported as an
import problem and the planner patches the wrong line. This module trains a multinomial
logistic regression over the TF-IDF/LSA embedding of the whole transcript, reports macro
F1 and expected calibration error on a stratified hold-out, and only replaces the rules
once the artifact proves itself.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .corpus import ALL_TRACEBACKS, LABELS, SEED_TRACEBACKS, is_hard
from .embeddings import HashingEmbedder, TfidfLsaEmbedder, build_embedder

log = logging.getLogger("self_healing.classifier")

RULE_ORDER = (
    ("syntaxerror", "syntax_error"),
    ("indentationerror", "indentation_error"),
    ("modulenotfounderror", "import_error"),
    ("importerror", "import_error"),
    ("nameerror", "name_error"),
    ("assertionerror", "assertion_failure"),
    ("typeerror", "type_error"),
)


def rule_class(text: str, *, exit_code: int = 1, timed_out: bool = False) -> str:
    """The shipped heuristic, kept here so both sides can be scored on one corpus."""
    if timed_out:
        return "timeout"
    low = (text or "").lower()
    if "timeouterror" in low or "timed out" in low:
        return "timeout"
    for needle, label in RULE_ORDER:
        if needle in low:
            return label
    if "error" in low or "failed" in low:
        return "runtime_error"
    return "ok" if exit_code == 0 else "runtime_error"


_EXC_NAME_RE = __import__("re").compile(r"\b([A-Z][A-Za-z]*(?:Error|Exception|Failure|Timeout))\b")


def structured_view(text: str) -> str:
    """Exception sequence as tokens: a bag of words cannot see that the *outermost*
    exception is the one printed last, which is what decides the label.

    Underscore-joined single tokens only -- the transcript tokenizer splits on "=" but
    keeps "_", and a separator between role and name would unbind the pair.
    """
    names = [n.lower() for n in _EXC_NAME_RE.findall(text or "")]
    if not names:
        return "exc_none"
    parts = [f"firstexc_{names[0]}", f"lastexc_{names[-1]}", f"nexc_{len(set(names))}"]
    parts += [f"anyexc_{n}" for n in sorted(set(names))]
    parts += [f"pair_{a}_{b}" for a, b in zip(names, names[1:])]
    if len(set(names)) == 1:
        parts.append(f"onlyexc_{names[0]}")
    return " ".join(parts)


def view(text: str, structured: bool = True) -> str:
    return f"{text}\n{structured_view(text)}" if structured else text


def _stratified_split(rows: Sequence[tuple[str, str]], *, test_size: float = 0.3, seed: int = 7):
    rng = np.random.default_rng(seed)
    train: list[tuple[str, str]] = []
    test: list[tuple[str, str]] = []
    for label in sorted({r[1] for r in rows}):
        group = [r for r in rows if r[1] == label]
        rng.shuffle(group)
        cut = int(round(len(group) * (1 - test_size)))
        cut = max(1, min(len(group) - 1, cut)) if len(group) > 1 else len(group)
        train.extend(group[:cut])
        test.extend(group[cut:])
    return train, test


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Confidence vs accuracy gap: an over-confident planner wastes repair attempts."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        total += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(total)


def classification_report(y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    per_class: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    support = sum(v["support"] for v in per_class.values())
    macro = sum(v["f1"] for v in per_class.values()) / len(per_class) if per_class else 0.0
    weighted = sum(v["f1"] * v["support"] for v in per_class.values()) / support if support else 0.0
    return {
        "accuracy": round(sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(len(y_true), 1), 4),
        "macro_f1": round(macro, 4),
        "weighted_f1": round(weighted, 4),
        "per_class": per_class,
    }


@dataclass
class TracebackClassifier:
    """Logistic regression on LSA features, with a centroid fallback that needs no sklearn.

    Calibration is on by default: at C=4 it lifted hold-out accuracy from 0.804 to 0.839
    and cut ECE from 0.290 to 0.206, whereas the best-raw-calibration setting (C=1) costs
    seven macro-F1 points. The planner thresholds P(green), so it needs both.
    """

    embedder: object = field(default_factory=lambda: HashingEmbedder(dim=192))
    labels: tuple[str, ...] = LABELS
    C: float = 4.0
    backend: str = "logistic"
    structured: bool = True
    calibrate: bool = True
    calibration: str = field(default="none", init=False)
    _model: object = field(default=None, repr=False, init=False)
    _centroids: dict[str, np.ndarray] = field(default_factory=dict, repr=False, init=False)
    metrics: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ training
    def fit(self, rows: Sequence[tuple[str, str]]) -> "TracebackClassifier":
        from sklearn.linear_model import LogisticRegression

        texts = [view(t, self.structured) for t, _ in rows]
        labels = [l for _, l in rows]
        if len(rows) < 8 or len(set(labels)) < 2:
            raise ValueError("need >=8 rows spanning >=2 classes")
        X = self.embedder.embed(texts)
        try:
            base = LogisticRegression(C=self.C, max_iter=2000, class_weight="balanced")
            model, self.calibration = self._maybe_calibrate(base, len(rows))
            model.fit(X, labels)
            self._model = model
            self.backend = "logistic"
        except Exception as exc:  # scipy/sklearn absent in a stripped-down runtime
            log.warning("logistic regression unavailable, falling back to centroids", extra={"error": str(exc)})
            self._fit_centroids(X, labels)
            self.backend = "centroid"
        return self

    def _maybe_calibrate(self, base, n_rows: int):
        """Planner decisions compare P(green) against a threshold, so the scores must
        mean frequencies, not just rank classes."""
        if not (self.calibrate and n_rows >= 24):
            return base, "none"
        try:
            from sklearn.calibration import CalibratedClassifierCV

            return CalibratedClassifierCV(base, method="sigmoid", cv=3), "sigmoid-cv3"
        except Exception as exc:  # pragma: no cover - calibration is an upgrade, not a requirement
            log.warning("calibration unavailable", extra={"error": str(exc)})
            return base, "none"

    def _fit_centroids(self, X: np.ndarray, labels: Sequence[str]) -> None:
        arr = np.asarray(labels)
        self._centroids = {
            label: X[arr == label].mean(axis=0) for label in sorted(set(labels))
        }

    def train_from_corpus(
        self,
        rows: Sequence[tuple[str, str]] | None = None,
        *,
        test_size: float = 0.3,
        seed: int = 7,
        save_to: Path | str | None = None,
    ) -> dict:
        """Fit on train, score on hold-out, keep the rules as a live baseline."""
        rows = list(rows or ALL_TRACEBACKS)
        train, test = _stratified_split(rows, test_size=test_size, seed=seed)
        self.fit(train)
        texts = [t for t, _ in test]
        y_true = [l for _, l in test]
        y_pred = [self.predict(t) for t in texts]
        rule_pred = [rule_class(t) for t in texts]
        probs = self.predict_proba(texts)
        report = classification_report(y_true, y_pred)
        label_idx = np.asarray([self.labels.index(y) for y in y_true])
        report["ece"] = round(expected_calibration_error(probs, label_idx), 4)
        report["rule_baseline_macro_f1"] = classification_report(y_true, rule_pred)["macro_f1"]

        hard = {i for i, t in enumerate(texts) if is_hard(t)}
        for name, picked in (("chained", sorted(hard)), ("single_exception", sorted(set(range(len(texts))) - hard))):
            sub = [y_true[i] for i in picked]
            report[f"{name}_n"] = len(picked)
            report[f"{name}_model_macro_f1"] = classification_report(sub, [y_pred[i] for i in picked])["macro_f1"] if sub else 0.0
            report[f"{name}_rule_macro_f1"] = classification_report(sub, [rule_pred[i] for i in picked])["macro_f1"] if sub else 0.0
        bag = self._bag_ablation(train, test)
        report["bag_only_macro_f1"] = bag["macro_f1"]
        report["bag_only_chained_macro_f1"] = bag["chained_macro_f1"]

        report["train_size"] = len(train)
        report["test_size"] = len(test)
        report["backend"] = self.backend
        report["calibration"] = self.calibration
        report["features"] = "tfidf-lsa+exception-order" if self.structured else "tfidf-lsa"
        report["classes"] = sorted(set(y_true))
        self.metrics = report
        if save_to:
            self.save(save_to)
        log.info(
            "classifier trained",
            extra={
                "macro_f1": report["macro_f1"],
                "chained_model": report["chained_model_macro_f1"],
                "chained_rules": report["chained_rule_macro_f1"],
                "ece": report["ece"],
            },
        )
        return report

    def _bag_ablation(self, train, test) -> dict:
        """Same split, same model, every order-sensitive feature switched off.

        The control has to be a real bag of words. Reusing the production vectorizer on
        raw text would keep word bigrams and char 3-5 grams, which already encode which
        exception precedes which - the comparison would measure nothing and could even
        flatter the control. `hashing` is already order-blind, so only tfidf-lsa needs a
        bespoke unigram fit.
        """
        try:
            corpus = [t for t, _ in train]
            dim = int(getattr(self.embedder, "dim", 64))
            if getattr(self.embedder, "name", "") == "tfidf-lsa":
                control = TfidfLsaEmbedder(dim=dim, word_ngram_range=(1, 1), char_ngram_range=None).fit(corpus)
            else:
                control = build_embedder(getattr(self.embedder, "name", "hashing"), corpus=corpus, dim=dim)
            shadow = TracebackClassifier(embedder=control, C=self.C, structured=False)
            shadow.fit(train)
            sub = [l for _, l in test]
            pred = [shadow.predict(t) for t, _ in test]
            chained = [i for i, (t, _) in enumerate(test) if is_hard(t)]
            return {
                "macro_f1": classification_report(sub, pred)["macro_f1"],
                "chained_macro_f1": classification_report([sub[i] for i in chained], [pred[i] for i in chained])["macro_f1"] if chained else 0.0,
            }
        except Exception as exc:  # pragma: no cover - ablation must never break training
            log.warning("bag-only ablation skipped", extra={"error": str(exc)})
            return {"macro_f1": 0.0, "chained_macro_f1": 0.0}

    # ------------------------------------------------------------------ inference
    def predict(self, text: str) -> str:
        return self.predict_one([text])[0]

    def predict_one(self, texts: Sequence[str]) -> list[str]:
        viewed = [view(t, self.structured) for t in texts]
        if self._model is not None:
            return [str(x) for x in self._model.predict(self.embedder.embed(viewed))]
        if not self._centroids:
            raise RuntimeError("classifier is not fitted")
        X = self.embedder.embed(viewed)
        table = np.vstack([self._centroids[k] for k in sorted(self._centroids)])
        names = sorted(self._centroids)
        return [names[int(i)] for i in np.argmax(X @ table.T, axis=1)]

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        viewed = [view(t, self.structured) for t in texts]
        if self._model is not None:
            return np.asarray(self._model.predict_proba(self.embedder.embed(viewed)), dtype=np.float64)
        X = self.embedder.embed(viewed)
        names = sorted(self._centroids)
        table = np.vstack([self._centroids[k] for k in names])
        sims = X @ table.T
        exp = np.exp(sims - sims.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def explain(self, text: str, top: int = 3) -> list[tuple[str, float]]:
        probs = self.predict_proba([text])[0]
        names = list(getattr(self._model, "classes_", names_of(self._centroids)))
        pairs = sorted(zip(names, probs), key=lambda kv: -kv[1])[:top]
        return [(str(n), round(float(p), 4)) for n, p in pairs]

    # ---------------------------------------------------------------- persistence
    def save(self, path: Path | str) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"embedder": self.embedder, "model": self._model, "centroids": self._centroids, "metrics": self.metrics, "labels": self.labels, "structured": self.structured},
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path | str) -> "TracebackClassifier":
        import joblib

        blob = joblib.load(Path(path))
        obj = cls(embedder=blob["embedder"], labels=tuple(blob["labels"]))
        obj.structured = bool(blob.get("structured", True))
        obj._model = blob["model"]
        obj._centroids = blob["centroids"]
        obj.metrics = blob.get("metrics", {})
        obj.backend = "logistic" if obj._model is not None else "centroid"
        return obj


def names_of(centroids: dict[str, np.ndarray]) -> list[str]:
    return sorted(centroids)


def build_classifier(
    *,
    model_path: Path | str | None = None,
    backend: str = "tfidf-lsa",
    dim: int = 96,
    auto_train: bool = True,
) -> TracebackClassifier | None:
    """Load the shipped artifact, or train one from the seed corpus and cache it."""
    path = Path(model_path) if model_path else None
    if path and path.exists():
        try:
            return TracebackClassifier.load(path)
        except Exception as exc:
            log.warning("classifier artifact unreadable, retraining", extra={"error": str(exc)})
    if not auto_train:
        return None
    classifier = TracebackClassifier(embedder=build_embedder(backend, corpus=[t for t, _ in ALL_TRACEBACKS], dim=dim))
    classifier.train_from_corpus(save_to=path)
    return classifier


def open_classifier(settings) -> TracebackClassifier | None:
    """Load the artifact named by Settings, and never train while answering.

    `build_classifier` auto-trains, which is right for an eval script and wrong for
    the repair loop: fitting costs seconds, and an unattended `shp heal` would write
    a model into the user's repo. Here a missing artifact is a normal state, not an
    error -- it just means the keyword rules keep the seat.
    """
    if not getattr(settings, "use_ml", False):
        return None
    path = Path(getattr(settings, "ml_model_path", ""))
    if not str(path) or not path.exists():
        log.warning("classifier artifact missing, using rules", extra={"path": str(path)})
        return None
    try:
        return TracebackClassifier.load(path)
    except Exception as exc:
        log.warning("classifier artifact unreadable, using rules", extra={"error": str(exc)})
        return None


def model_card(classifier: TracebackClassifier) -> str:
    """Human-readable provenance, the thing reviewers open first."""
    m = classifier.metrics or {}
    lines = [
        "# Failure classifier model card",
        "",
        f"- backend: `{classifier.backend}` over `{classifier.embedder.name}` embeddings (dim {getattr(classifier.embedder, 'dim', '?')})",
        f"- classes: {len(classifier.labels)}",
        f"- train/test: {m.get('train_size', '?')}/{m.get('test_size', '?')} stratified split",
        f"- accuracy **{m.get('accuracy', '?')}**, macro-F1 **{m.get('macro_f1', '?')}** (rule baseline {m.get('rule_baseline_macro_f1', '?')})",
        f"- expected calibration error: {m.get('ece', '?')}",
        f"- chained tracebacks (n={m.get('chained_n', 0)}): model {m.get('chained_model_macro_f1', '?')} vs rules {m.get('chained_rule_macro_f1', '?')}",
        f"- single-exception traces (n={m.get('single_exception_n', 0)}): model {m.get('single_exception_model_macro_f1', '?')} vs rules {m.get('single_exception_rule_macro_f1', '?')}",
        f"- ablation: exception-order features move macro-F1 from {m.get('bag_only_macro_f1', '?')} (bag of words) to {m.get('macro_f1', '?')}",
        "",
        "| class | precision | recall | f1 | support |",
        "|---|---|---|---|---|",
    ]
    for label, row in sorted((m.get("per_class") or {}).items()):
        lines.append(f"| `{label}` | {row['precision']} | {row['recall']} | {row['f1']} | {row['support']} |")
    return "\n".join(lines) + "\n"
