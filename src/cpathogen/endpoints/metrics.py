"""Endpoint metrics with explicit classification and survival semantics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def multiclass_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Return per-class one-vs-rest and macro multiclass metrics."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    classes = list(classes)
    if probabilities.shape != (len(labels), len(classes)):
        raise ValueError("Probability matrix shape does not match labels/classes")
    predictions = probabilities.argmax(axis=1)
    per_class: dict[str, dict[str, float | int]] = {}
    aucs: list[float] = []
    f1s: list[float] = []
    class_accuracies: list[float] = []
    for index, name in enumerate(classes):
        truth = labels == index
        predicted = predictions == index
        auc = float(roc_auc_score(truth.astype(int), probabilities[:, index]))
        f1 = float(f1_score(truth, predicted, zero_division=0))
        class_accuracy = float(np.mean(predictions[truth] == index))
        per_class[name] = {
            "roc_auc_ovr": auc,
            "f1_ovr": f1,
            # This is recall for the class; the explicit name avoids claiming
            # that one-vs-rest accuracy is a useful imbalance-aware metric.
            "class_accuracy": class_accuracy,
            "support": int(truth.sum()),
        }
        aucs.append(auc)
        f1s.append(f1)
        class_accuracies.append(class_accuracy)
    return {
        "overall_accuracy": float(accuracy_score(labels, predictions)),
        "macro_roc_auc_ovr": float(np.mean(aucs)),
        "macro_f1": float(np.mean(f1s)),
        "macro_class_accuracy": float(np.mean(class_accuracies)),
        "per_class": per_class,
        "class_accuracy_definition": (
            "Per-class accuracy is class recall (correct predictions among true members); "
            "overall_accuracy is reported separately."
        ),
        "c_index": None,
        "c_index_note": "C-index is not defined per PAM50 class.",
    }


def concordance_index(
    event_time: np.ndarray,
    event_observed: np.ndarray,
    risk: np.ndarray,
) -> float:
    """Harrell's C-index for higher-risk-means-earlier-event predictions."""
    time = np.asarray(event_time, dtype=float)
    event = np.asarray(event_observed, dtype=bool)
    risk = np.asarray(risk, dtype=float)
    if not (len(time) == len(event) == len(risk)):
        raise ValueError("Survival arrays have different lengths")
    concordant = 0.0
    comparable = 0
    for first in range(len(time)):
        for second in range(first + 1, len(time)):
            if time[first] == time[second]:
                continue
            if time[first] < time[second] and event[first]:
                earlier, later = first, second
            elif time[second] < time[first] and event[second]:
                earlier, later = second, first
            else:
                continue
            comparable += 1
            if risk[earlier] > risk[later]:
                concordant += 1.0
            elif risk[earlier] == risk[later]:
                concordant += 0.5
    if comparable == 0:
        raise ValueError("No comparable survival pairs")
    return concordant / comparable


def known_horizon_status(
    event_time: np.ndarray,
    event_observed: np.ndarray,
    horizon_days: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return known event-by-horizon labels and their inclusion mask."""
    time = np.asarray(event_time, dtype=float)
    event = np.asarray(event_observed, dtype=bool)
    event_by_horizon = event & (time <= horizon_days)
    known = event_by_horizon | (time > horizon_days)
    return event_by_horizon.astype(int), known
