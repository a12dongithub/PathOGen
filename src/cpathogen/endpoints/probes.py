"""Patient-level cross-fitted PAM50 and survival probes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .clinical import PAM50_CLASSES
from .cox import CoxProbe, fit_cox_probe
from .metrics import concordance_index, known_horizon_status, multiclass_metrics


def aggregate_patient_embeddings(
    tile_manifest: pd.DataFrame, tile_embeddings: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    """Mean-pool normalized tile features so every patient is one sample."""
    if len(tile_manifest) != len(tile_embeddings):
        raise ValueError("Manifest and embedding rows differ")
    vectors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for patient, indices in tile_manifest.groupby(
        "patient_id", sort=True
    ).indices.items():
        pooled = tile_embeddings[np.asarray(indices)].mean(axis=0)
        norm = np.linalg.norm(pooled)
        if norm > 0:
            pooled = pooled / norm
        vectors.append(pooled.astype(np.float32, copy=False))
        rows.append({"patient_id": patient, "tile_count": len(indices)})
    return pd.DataFrame(rows), np.stack(vectors)


def assign_pam50_folds(
    patients: pd.DataFrame, *, n_folds: int, seed: int
) -> np.ndarray:
    labels = patients["pam50"].map({name: i for i, name in enumerate(PAM50_CLASSES)})
    if labels.isna().any():
        raise ValueError("PAM50 fold input contains missing or unsupported labels")
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = np.full(len(patients), -1, dtype=np.int64)
    for fold, (_, test) in enumerate(splitter.split(np.zeros(len(patients)), labels)):
        folds[test] = fold
    return folds


def assign_survival_folds(
    patients: pd.DataFrame, *, n_folds: int, seed: int
) -> np.ndarray:
    """Stratify by event and coarse time bins while retaining patient-level splits."""
    event = patients["survival_event"].astype(int)
    time = patients["survival_time_days"].astype(float)
    try:
        bins = pd.qcut(time.rank(method="first"), q=min(4, n_folds), labels=False)
        strata = event.astype(str) + "_" + bins.astype(str)
        if strata.value_counts().min() < n_folds:
            raise ValueError
    except ValueError:
        strata = event.astype(str)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = np.full(len(patients), -1, dtype=np.int64)
    for fold, (_, test) in enumerate(splitter.split(np.zeros(len(patients)), strata)):
        folds[test] = fold
    return folds


def fit_pam50_crossfit(
    features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    *,
    seed: int,
    c_value: float,
) -> tuple[dict[int, Any], np.ndarray, dict[str, Any]]:
    """Fit class-balanced multinomial heads and return patient-level OOF scores."""
    probabilities = np.full((len(labels), len(PAM50_CLASSES)), np.nan, dtype=float)
    heads: dict[int, Any] = {}
    fold_metrics = []
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        head = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=3_000,
                random_state=seed,
            ),
        )
        head.fit(features[train], labels[train])
        predicted = head.predict_proba(features[test])
        class_order = head[-1].classes_.astype(int)
        probabilities[np.ix_(test, class_order)] = predicted
        heads[int(fold)] = head
        fold_metrics.append(
            {
                "fold": int(fold),
                "train_patients": int(train.sum()),
                "test_patients": int(test.sum()),
                "metrics": multiclass_metrics(
                    labels[test], probabilities[test], PAM50_CLASSES
                ),
            }
        )
    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            "Some PAM50 patients did not receive complete OOF probabilities"
        )
    return (
        heads,
        probabilities,
        {
            "overall_oof": multiclass_metrics(labels, probabilities, PAM50_CLASSES),
            "folds": fold_metrics,
        },
    )


def _horizon_metrics(
    time: np.ndarray,
    event: np.ndarray,
    event_probability: np.ndarray,
    horizon_days: int,
) -> dict[str, Any]:
    labels, known = known_horizon_status(time, event, horizon_days)
    if known.sum() == 0 or len(np.unique(labels[known])) < 2:
        return {
            "horizon_days": horizon_days,
            "known_patients": int(known.sum()),
            "evaluable": False,
        }
    prediction = event_probability[known] >= 0.5
    return {
        "horizon_days": horizon_days,
        "known_patients": int(known.sum()),
        "evaluable": True,
        "roc_auc": float(roc_auc_score(labels[known], event_probability[known])),
        "f1": float(f1_score(labels[known], prediction, zero_division=0)),
        "accuracy": float(accuracy_score(labels[known], prediction)),
        "note": "Censored before the horizon are excluded; 0.5 is the fixed probability threshold.",
    }


def fit_survival_crossfit(
    features: np.ndarray,
    event_time: np.ndarray,
    event_observed: np.ndarray,
    folds: np.ndarray,
    *,
    seed: int,
    pca_components: int,
    l2_penalty: float,
    horizons_days: tuple[int, ...] = (1826, 3652),
) -> tuple[dict[int, CoxProbe], pd.DataFrame, dict[str, Any]]:
    """Fit censored Cox heads and return held-out risk and survival estimates."""
    risk = np.full(len(features), np.nan, dtype=float)
    survival = np.full((len(features), len(horizons_days)), np.nan, dtype=float)
    heads: dict[int, CoxProbe] = {}
    fold_results = []
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        test = folds == fold
        head = fit_cox_probe(
            features[train],
            event_time[train],
            event_observed[train],
            pca_components=pca_components,
            l2_penalty=l2_penalty,
            seed=seed,
        )
        risk[test] = head.predict_risk(features[test])
        survival[test] = head.predict_survival(
            features[test], np.asarray(horizons_days)
        )
        heads[int(fold)] = head
        try:
            fold_c_index = concordance_index(
                event_time[test], event_observed[test], risk[test]
            )
        except ValueError:
            fold_c_index = None
        fold_results.append(
            {
                "fold": int(fold),
                "train_patients": int(train.sum()),
                "test_patients": int(test.sum()),
                "events_test": int(event_observed[test].sum()),
                "c_index": fold_c_index,
            }
        )
    if not np.isfinite(risk).all() or not np.isfinite(survival).all():
        raise RuntimeError("Some survival patients did not receive OOF predictions")
    prediction_frame = pd.DataFrame({"risk_oof": risk})
    horizon_results = []
    for index, horizon in enumerate(horizons_days):
        prediction_frame[f"survival_probability_{horizon}d_oof"] = survival[:, index]
        prediction_frame[f"event_probability_{horizon}d_oof"] = 1.0 - survival[:, index]
        horizon_results.append(
            _horizon_metrics(
                event_time,
                event_observed,
                1.0 - survival[:, index],
                horizon,
            )
        )
    return (
        heads,
        prediction_frame,
        {
            "overall_oof_c_index": concordance_index(event_time, event_observed, risk),
            "folds": fold_results,
            "horizon_classification": horizon_results,
            "metric_note": "Primary survival metric is Harrell's C-index; horizon metrics are secondary.",
        },
    )
