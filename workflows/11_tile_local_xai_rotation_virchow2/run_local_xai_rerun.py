#!/usr/bin/env python3
"""Recompute CPathOGen endpoint sensitivity without variable patient-bag dilution.

PAM50 uses patient-disjoint, patient-balanced tile-level logistic probes.  Each
generated baseline/counterfactual pair is scored directly, so its TVD is local
to that tile.  Survival remains a patient endpoint: Cox probes are retrained on
deterministic fixed-size bags, and every generated variant replaces one slot in
an otherwise identical fixed-size context bag.

The script consumes the embedding caches saved by the original Colab run.  It
does not regenerate images or download encoders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

PAM50_CLASSES = ("Basal", "HER2", "LumA", "LumB")
HORIZON_DAYS = 1826.0
DEFAULT_MODELS = ("resnet50", "ctranspath", "uni2h")
SUPPORTED_MODELS = (*DEFAULT_MODELS, "virchow2")
DISPLAY_MODELS = {
    "resnet50": "ResNet-50",
    "ctranspath": "CTransPath",
    "uni2h": "UNI2-h",
    "virchow2": "Virchow2",
    "pathlupi_conch": "PathLUPI + CONCH",
}


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    display_name: str
    family: str
    reference: str
    targets: tuple[str, ...]


EXPERIMENTS = (
    Experiment(
        "stain_brightness",
        "Stain brightness",
        "nuisance",
        "baseline",
        (
            "stain_brightness_minus_2p0sd",
            "stain_brightness_minus_1p0sd",
            "stain_brightness_plus_1p0sd",
            "stain_brightness_plus_2p0sd",
        ),
    ),
    Experiment(
        "image_rotation",
        "Image rotation",
        "nuisance",
        "rotation_0",
        ("rotation_90", "rotation_180", "rotation_270"),
    ),
    Experiment(
        "nuclear_enlargement",
        "Nuclear enlargement",
        "biological",
        "baseline",
        (
            "nuclear_enlargement_minus_2p0sd",
            "nuclear_enlargement_minus_1p0sd",
            "nuclear_enlargement_plus_1p0sd",
            "nuclear_enlargement_plus_2p0sd",
        ),
    ),
    Experiment(
        "nuclear_shape_irregularity",
        "Shape irregularity",
        "biological",
        "baseline",
        (
            "nuclear_shape_irregularity_minus_2p0sd",
            "nuclear_shape_irregularity_minus_1p0sd",
            "nuclear_shape_irregularity_plus_1p0sd",
            "nuclear_shape_irregularity_plus_2p0sd",
        ),
    ),
    Experiment(
        "peritumoral_immune_ring_diameter40px",
        "Immune burden",
        "biological",
        "peritumoral_ring_plus_80",
        ("peritumoral_ring_plus_160", "peritumoral_ring_plus_320"),
    ),
    Experiment(
        "tumor_immune_separation_diameter40px",
        "Tumor-immune mixing",
        "biological",
        "tumor_immune_maximal_mixing",
        (
            "tumor_immune_low_separation",
            "tumor_immune_intermediate",
            "tumor_immune_high_separation",
            "tumor_immune_maximal_segregation",
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-root", type=Path, required=True)
    parser.add_argument("--counterfactual-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--bag-sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--primary-bag-size", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pam50-c", type=float, default=1.0)
    parser.add_argument("--pam50-max-iter", type=int, default=400)
    parser.add_argument("--max-train-tiles-per-patient", type=int, default=64)
    parser.add_argument("--survival-pca-components", type=int, default=64)
    parser.add_argument("--survival-l2", type=float, default=1.0)
    parser.add_argument(
        "--max-panels",
        type=int,
        help="Limit source panels per experiment for a smoke test.",
    )
    return parser.parse_args()


def l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def stable_rng(seed: int, *parts: str) -> np.random.Generator:
    payload = "|".join((str(seed), *map(str, parts))).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def load_cache(
    endpoint_root: Path,
    model: str,
    tile_manifest: pd.DataFrame,
    variant_manifest: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    cache_root = endpoint_root / "embedding_cache"
    tile_cache = np.load(cache_root / f"{model}_tiles.npz", allow_pickle=False)
    variant_cache = np.load(
        cache_root / f"{model}_counterfactuals.npz", allow_pickle=False
    )
    tile_ids = tile_cache["tile_ids"].astype(str)
    variant_ids = variant_cache["variant_ids"].astype(str)
    expected_tiles = tile_manifest["tile_id"].astype(str).to_numpy()
    expected_variants = variant_manifest["variant_id"].astype(str).to_numpy()
    if not np.array_equal(tile_ids, expected_tiles):
        raise ValueError(f"{model}: baseline cache order does not match tile manifest")
    if not np.array_equal(variant_ids, expected_variants):
        raise ValueError(
            f"{model}: counterfactual cache order does not match variant manifest"
        )
    tile_values = tile_cache["embeddings"].astype(np.float32, copy=False)
    variant_values = variant_cache["embeddings"].astype(np.float32, copy=False)
    if not np.isfinite(tile_values).all() or not np.isfinite(variant_values).all():
        raise ValueError(f"{model}: embedding cache contains non-finite values")
    return l2_normalize(tile_values), l2_normalize(variant_values)


def validate_local_images(
    variant_manifest: pd.DataFrame, counterfactual_root: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = variant_manifest.copy()
    actual_images = {
        str(path.resolve()).lower() for path in counterfactual_root.rglob("*.png")
    }
    local_paths = []
    exists = []
    for row in result.itertuples(index=False):
        path = (
            counterfactual_root
            / str(row.experiment)
            / str(row.source_tile_id)
            / f"{row.condition}.png"
        )
        local_paths.append(str(path))
        exists.append(str(path.resolve()).lower() in actual_images)
    result["local_image_path"] = local_paths
    result["local_image_exists"] = exists
    return result, {
        "manifest_variants": len(result),
        "local_images_found": int(np.sum(exists)),
        "local_images_missing": int(len(result) - np.sum(exists)),
    }


def load_fold_map(path: Path) -> dict[str, int]:
    frame = pd.read_csv(path)
    return dict(zip(frame["patient_id"].astype(str), frame["outer_fold"].astype(int)))


def patient_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    patient_counts = frame["patient_id"].value_counts()
    patient_labels = frame[["patient_id", "class_index"]].drop_duplicates()
    class_counts = patient_labels["class_index"].value_counts()
    class_factor = {
        label: len(patient_labels) / (len(PAM50_CLASSES) * count)
        for label, count in class_counts.items()
    }
    weights = np.asarray(
        [
            class_factor[int(row.class_index)] / patient_counts[str(row.patient_id)]
            for row in frame.itertuples(index=False)
        ],
        dtype=np.float64,
    )
    return weights * (len(weights) / weights.sum())


def cap_training_tiles(
    frame: pd.DataFrame, maximum: int, seed: int
) -> pd.DataFrame:
    if maximum <= 0:
        return frame
    selected = []
    for patient, group in frame.groupby("patient_id", sort=True):
        indices = group.index.to_numpy()
        if len(indices) > maximum:
            indices = stable_rng(seed, "pam50", patient).choice(
                indices, size=maximum, replace=False
            )
        selected.extend(indices.tolist())
    return frame.loc[sorted(selected)].copy()


def fit_tile_pam50(
    tile_manifest: pd.DataFrame,
    clinical: pd.DataFrame,
    embeddings: np.ndarray,
    fold_map: dict[str, int],
    *,
    c_value: float,
    max_iter: int,
    max_tiles_per_patient: int,
    seed: int,
) -> tuple[dict[int, LogisticRegression], dict[str, Any], pd.DataFrame]:
    label_map = dict(zip(clinical["patient_id"], clinical["pam50"]))
    frame = tile_manifest[["tile_id", "patient_id"]].copy()
    frame["pam50"] = frame["patient_id"].map(label_map)
    frame["outer_fold"] = frame["patient_id"].map(fold_map)
    frame = frame[
        frame["pam50"].isin(PAM50_CLASSES) & frame["outer_fold"].notna()
    ].copy()
    frame["outer_fold"] = frame["outer_fold"].astype(int)
    frame["class_index"] = frame["pam50"].map(
        {name: index for index, name in enumerate(PAM50_CLASSES)}
    )
    frame["embedding_index"] = frame.index.astype(int)
    oof = np.full((len(frame), len(PAM50_CLASSES)), np.nan, dtype=np.float64)
    frame = frame.reset_index(drop=True)
    heads: dict[int, LogisticRegression] = {}
    fit_rows = []
    for fold in sorted(frame["outer_fold"].unique()):
        train = frame[frame["outer_fold"] != fold]
        train = cap_training_tiles(train, max_tiles_per_patient, seed + fold)
        test = frame[frame["outer_fold"] == fold]
        weights = patient_balanced_weights(train)
        head = LogisticRegression(
            C=c_value,
            l1_ratio=0,
            solver="lbfgs",
            max_iter=max_iter,
            tol=1e-4,
            random_state=seed + fold,
        )
        head.fit(
            embeddings[train["embedding_index"].to_numpy()],
            train["class_index"].to_numpy(),
            sample_weight=weights,
        )
        probabilities = head.predict_proba(
            embeddings[test["embedding_index"].to_numpy()]
        )
        oof[test.index.to_numpy()] = probabilities
        heads[int(fold)] = head
        fit_rows.append(
            {
                "fold": int(fold),
                "training_tiles": len(train),
                "training_patients": int(train["patient_id"].nunique()),
                "test_tiles": len(test),
                "iterations": int(np.max(head.n_iter_)),
            }
        )
    if not np.isfinite(oof).all():
        raise RuntimeError("PAM50 OOF probabilities are incomplete")
    for index, name in enumerate(PAM50_CLASSES):
        frame[f"probability_{name}"] = oof[:, index]
    patient_probabilities = (
        frame.groupby(["patient_id", "pam50"], as_index=False)[
            [f"probability_{name}" for name in PAM50_CLASSES]
        ]
        .mean()
        .sort_values("patient_id", kind="stable")
    )
    labels = patient_probabilities["pam50"].map(
        {name: index for index, name in enumerate(PAM50_CLASSES)}
    ).to_numpy()
    probabilities = patient_probabilities[
        [f"probability_{name}" for name in PAM50_CLASSES]
    ].to_numpy()
    predictions = probabilities.argmax(axis=1)
    metrics = {
        "unit_for_head": "tile",
        "split": "patient-disjoint cross-fitting",
        "training_weighting": "equal patient mass and equal PAM50 class mass",
        "patients": len(patient_probabilities),
        "tiles": len(frame),
        "patient_macro_auc_ovr": float(
            roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
        ),
        "patient_macro_f1": float(f1_score(labels, predictions, average="macro")),
        "patient_accuracy": float(accuracy_score(labels, predictions)),
        "folds": fit_rows,
    }
    return heads, metrics, patient_probabilities


@dataclass
class CoxProbe:
    scaler: StandardScaler
    pca: PCA
    coefficient: np.ndarray
    event_times: np.ndarray
    cumulative_hazard: np.ndarray

    def transform(self, features: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(features))

    def predict_risk(self, features: np.ndarray) -> np.ndarray:
        return self.transform(features) @ self.coefficient

    def predict_survival(self, features: np.ndarray, horizon: float) -> np.ndarray:
        index = np.searchsorted(self.event_times, horizon, side="right") - 1
        baseline = 0.0 if index < 0 else float(self.cumulative_hazard[index])
        relative_risk = np.exp(np.clip(self.predict_risk(features), -30, 30))
        return np.exp(-relative_risk * baseline)


def fit_cox(
    features: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    *,
    pca_components: int,
    l2_penalty: float,
    seed: int,
) -> CoxProbe:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    components = max(1, min(pca_components, len(features) - 1, features.shape[1]))
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    reduced = pca.fit_transform(scaled)
    event_times = np.unique(times[events.astype(bool)])
    event_groups = [np.flatnonzero((events == 1) & (times == t)) for t in event_times]
    risk_groups = [np.flatnonzero(times >= t) for t in event_times]
    event_count = int(events.sum())
    if event_count < 2:
        raise ValueError("At least two survival events are required")

    def objective(coefficient: np.ndarray) -> tuple[float, np.ndarray]:
        linear = reduced @ coefficient
        loss = 0.0
        gradient = np.zeros_like(coefficient)
        for event_indices, risk_indices in zip(event_groups, risk_groups, strict=True):
            risk_linear = linear[risk_indices]
            log_denominator = logsumexp(risk_linear)
            weights = np.exp(risk_linear - log_denominator)
            ties = len(event_indices)
            loss += ties * log_denominator - linear[event_indices].sum()
            gradient += ties * (weights @ reduced[risk_indices])
            gradient -= reduced[event_indices].sum(axis=0)
        loss = loss / event_count + 0.5 * l2_penalty * coefficient.dot(coefficient)
        gradient = gradient / event_count + l2_penalty * coefficient
        return float(loss), gradient

    result = minimize(
        objective,
        np.zeros(components, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Cox optimization failed: {result.message}")
    coefficient = np.asarray(result.x, dtype=np.float64)
    linear = reduced @ coefficient
    increments = []
    for event_indices, risk_indices in zip(event_groups, risk_groups, strict=True):
        denominator = np.exp(np.clip(linear[risk_indices], -30, 30)).sum()
        increments.append(len(event_indices) / denominator)
    return CoxProbe(
        scaler=scaler,
        pca=pca,
        coefficient=coefficient,
        event_times=event_times,
        cumulative_hazard=np.cumsum(np.asarray(increments, dtype=np.float64)),
    )


def concordance_index(times: np.ndarray, events: np.ndarray, risks: np.ndarray) -> float:
    concordant = 0.0
    comparable = 0
    for first in range(len(times)):
        for second in range(first + 1, len(times)):
            if times[first] == times[second]:
                continue
            if times[first] < times[second] and events[first]:
                earlier, later = first, second
            elif times[second] < times[first] and events[second]:
                earlier, later = second, first
            else:
                continue
            comparable += 1
            if risks[earlier] > risks[later]:
                concordant += 1.0
            elif risks[earlier] == risks[later]:
                concordant += 0.5
    if comparable == 0:
        raise ValueError("No comparable survival pairs")
    return concordant / comparable


def patient_index_groups(tile_manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        str(patient): np.asarray(indices, dtype=int)
        for patient, indices in tile_manifest.groupby("patient_id", sort=True).indices.items()
    }


def fixed_training_bags(
    groups: dict[str, np.ndarray],
    embeddings: np.ndarray,
    bag_size: int,
    seed: int,
) -> tuple[list[str], np.ndarray]:
    patients = []
    pooled = []
    for patient, indices in groups.items():
        if len(indices) < bag_size:
            continue
        selected = stable_rng(seed, "survival-train", patient, str(bag_size)).choice(
            indices, size=bag_size, replace=False
        )
        patients.append(patient)
        pooled.append(embeddings[selected].mean(axis=0))
    return patients, l2_normalize(np.asarray(pooled, dtype=np.float32))


def fit_fixed_bag_survival(
    tile_manifest: pd.DataFrame,
    clinical: pd.DataFrame,
    embeddings: np.ndarray,
    fold_map: dict[str, int],
    *,
    bag_size: int,
    pca_components: int,
    l2_penalty: float,
    seed: int,
) -> tuple[dict[int, CoxProbe], dict[str, Any], pd.DataFrame]:
    patients, features = fixed_training_bags(
        patient_index_groups(tile_manifest), embeddings, bag_size, seed
    )
    frame = pd.DataFrame({"patient_id": patients, "feature_index": range(len(patients))})
    frame = frame.merge(clinical, on="patient_id", how="left", validate="one_to_one")
    frame["outer_fold"] = frame["patient_id"].map(fold_map)
    frame = frame[
        frame[["survival_time_days", "survival_event", "outer_fold"]]
        .notna()
        .all(axis=1)
    ].copy()
    frame["outer_fold"] = frame["outer_fold"].astype(int)
    risks = np.full(len(frame), np.nan, dtype=np.float64)
    survival_5y = np.full(len(frame), np.nan, dtype=np.float64)
    frame = frame.reset_index(drop=True)
    heads: dict[int, CoxProbe] = {}
    fit_rows = []
    for fold in sorted(frame["outer_fold"].unique()):
        train = frame[frame["outer_fold"] != fold]
        test = frame[frame["outer_fold"] == fold]
        head = fit_cox(
            features[train["feature_index"].to_numpy()],
            train["survival_time_days"].to_numpy(float),
            train["survival_event"].to_numpy(int),
            pca_components=pca_components,
            l2_penalty=l2_penalty,
            seed=seed + fold,
        )
        test_features = features[test["feature_index"].to_numpy()]
        risks[test.index] = head.predict_risk(test_features)
        survival_5y[test.index] = head.predict_survival(test_features, HORIZON_DAYS)
        heads[int(fold)] = head
        fit_rows.append(
            {
                "fold": int(fold),
                "training_patients": len(train),
                "test_patients": len(test),
            }
        )
    if not np.isfinite(risks).all() or not np.isfinite(survival_5y).all():
        raise RuntimeError("Survival OOF predictions are incomplete")
    frame["risk_score"] = risks
    frame["survival_probability_5y"] = survival_5y
    metrics = {
        "unit": f"patient fixed bag of {bag_size} tiles",
        "patients": len(frame),
        "events": int(frame["survival_event"].sum()),
        "c_index": float(
            concordance_index(
                frame["survival_time_days"].to_numpy(float),
                frame["survival_event"].to_numpy(int),
                risks,
            )
        ),
        "folds": fit_rows,
    }
    return heads, metrics, frame


def predict_tile_variants(
    variants: pd.DataFrame,
    embeddings: np.ndarray,
    heads: dict[int, LogisticRegression],
    fold_map: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.full((len(variants), len(PAM50_CLASSES)), np.nan, dtype=np.float64)
    valid = np.zeros(len(variants), dtype=bool)
    folds = variants["patient_id_resolved"].map(fold_map)
    for fold, head in heads.items():
        indices = np.flatnonzero(folds.eq(fold).to_numpy())
        if len(indices):
            probabilities[indices] = head.predict_proba(embeddings[indices])
            valid[indices] = True
    # Patients without a PAM50 label were absent from every head's training
    # data.  They can therefore be scored by the cross-fold ensemble without
    # leaking an endpoint label or discarding an otherwise valid tile panel.
    unseen = variants["patient_id_resolved"].notna().to_numpy() & ~valid
    unseen_indices = np.flatnonzero(unseen)
    if len(unseen_indices):
        probabilities[unseen_indices] = np.mean(
            [head.predict_proba(embeddings[unseen_indices]) for head in heads.values()],
            axis=0,
        )
        valid[unseen_indices] = True
    return probabilities, valid


def predict_fixed_bag_variants(
    variants: pd.DataFrame,
    variant_embeddings: np.ndarray,
    tile_manifest: pd.DataFrame,
    tile_embeddings: np.ndarray,
    heads: dict[int, CoxProbe],
    fold_map: dict[str, int],
    *,
    bag_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.full(len(variants), np.nan, dtype=np.float64)
    valid = np.zeros(len(variants), dtype=bool)
    groups = patient_index_groups(tile_manifest)
    tile_lookup = {
        str(tile_id): index
        for index, tile_id in enumerate(tile_manifest["tile_id"].astype(str))
    }
    for source_tile, rows in variants.groupby("source_tile_id", sort=False):
        patient = str(rows["patient_id_resolved"].iloc[0])
        fold = fold_map.get(patient)
        source_index = tile_lookup.get(str(source_tile))
        patient_indices = groups.get(patient)
        if source_index is None or patient_indices is None:
            continue
        context_candidates = patient_indices[patient_indices != source_index]
        if len(context_candidates) < bag_size - 1:
            continue
        context = stable_rng(
            seed, "survival-score", patient, str(source_tile), str(bag_size)
        ).choice(context_candidates, size=bag_size - 1, replace=False)
        context_sum = tile_embeddings[context].sum(axis=0)
        row_indices = rows.index.to_numpy(dtype=int)
        pooled = l2_normalize(context_sum[None, :] + variant_embeddings[row_indices])
        if fold in heads:
            predictions[row_indices] = heads[int(fold)].predict_survival(
                pooled, HORIZON_DAYS
            )
        else:
            # No survival label means the patient was absent from all fitted
            # heads.  Average their probabilities to retain the XAI panel.
            predictions[row_indices] = np.mean(
                [head.predict_survival(pooled, HORIZON_DAYS) for head in heads.values()],
                axis=0,
            )
        valid[row_indices] = True
    return predictions, valid


def summarize_experiments(
    variants: pd.DataFrame,
    probabilities: np.ndarray,
    valid: np.ndarray,
    *,
    model: str,
    endpoint: str,
    bag_size: int | None,
    max_panels: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    summary_rows = []
    for spec in EXPERIMENTS:
        experiment = variants[
            variants["experiment"].eq(spec.experiment_id) & valid
        ]
        panel_rows = []
        groups = list(experiment.groupby("source_tile_id", sort=True))
        if max_panels:
            groups = groups[:max_panels]
        for source_tile, group in groups:
            references = group[group["condition"].eq(spec.reference)]
            targets = group[group["condition"].isin(spec.targets)]
            if len(references) != 1 or set(targets["condition"]) != set(spec.targets):
                continue
            reference_index = int(references.index[0])
            reference = probabilities[reference_index]
            reference_decision = int(np.argmax(reference)) if endpoint == "PAM50" else int(reference[0] >= 0.5)
            target_metrics = []
            for target in targets.itertuples():
                target_index = int(target.Index)
                target_probability = probabilities[target_index]
                target_decision = int(np.argmax(target_probability)) if endpoint == "PAM50" else int(target_probability[0] >= 0.5)
                tvd = float(0.5 * np.abs(reference - target_probability).sum())
                flip = float(target_decision != reference_decision)
                target_metrics.append((tvd, flip))
                detail_rows.append(
                    {
                        "model_id": model,
                        "model": DISPLAY_MODELS[model],
                        "endpoint": endpoint,
                        "bag_size": bag_size,
                        "experiment": spec.experiment_id,
                        "display_experiment": spec.display_name,
                        "family": spec.family,
                        "source_tile_id": source_tile,
                        "patient_id": target.patient_id_resolved,
                        "reference_condition": spec.reference,
                        "target_condition": target.condition,
                        "tvd": tvd,
                        "flip": flip,
                    }
                )
            values = np.asarray(target_metrics, dtype=float)
            panel_rows.append(
                {
                    "source_tile_id": source_tile,
                    "patient_id": str(group["patient_id_resolved"].iloc[0]),
                    "tvd": float(values[:, 0].mean()),
                    "flip": float(values[:, 1].mean()),
                }
            )
        panel = pd.DataFrame(panel_rows)
        if panel.empty:
            continue
        summary_rows.append(
            {
                "model_id": model,
                "model": DISPLAY_MODELS[model],
                "endpoint": endpoint,
                "bag_size": bag_size,
                "experiment": spec.experiment_id,
                "display_experiment": spec.display_name,
                "family": spec.family,
                "tiles": len(panel),
                "patients": int(panel["patient_id"].nunique()),
                "mean_tvd": float(panel["tvd"].mean()),
                "flip_rate": float(panel["flip"].mean()),
                "median_tvd": float(panel["tvd"].median()),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def bernoulli_vectors(values: np.ndarray) -> np.ndarray:
    return np.column_stack((values, 1.0 - values))


def bnr_for(summary: pd.DataFrame) -> float:
    nuisance = summary.loc[summary["family"].eq("nuisance"), "mean_tvd"]
    biological = summary.loc[summary["family"].eq("biological"), "mean_tvd"]
    if nuisance.empty or len(biological) != 4 or nuisance.mean() <= 0:
        return math.nan
    return float(biological.mean() / nuisance.mean())


def build_table(
    summaries: pd.DataFrame,
    performances: pd.DataFrame,
    primary_bag_size: int,
    model_order: tuple[str, ...] = DEFAULT_MODELS,
) -> pd.DataFrame:
    rows = []
    for endpoint in ("PAM50", "Overall survival"):
        for model in model_order:
            selected = summaries[
                summaries["model_id"].eq(model)
                & summaries["endpoint"].eq(endpoint)
            ]
            if endpoint == "PAM50":
                selected = selected[selected["bag_size"].isna()]
                metric_name = "patient_macro_auc_ovr"
            else:
                selected = selected[selected["bag_size"].eq(primary_bag_size)]
                metric_name = "c_index"
            if selected.empty:
                continue
            performance = performances[
                performances["model_id"].eq(model)
                & performances["endpoint"].eq(endpoint)
                & (
                    performances["bag_size"].isna()
                    if endpoint == "PAM50"
                    else performances["bag_size"].eq(primary_bag_size)
                )
            ]
            metric = float(performance.iloc[0][metric_name])
            row: dict[str, Any] = {
                "Task": endpoint,
                "Model": DISPLAY_MODELS[model],
                "Performance": metric,
            }
            for spec in EXPERIMENTS:
                match = selected[selected["experiment"].eq(spec.experiment_id)]
                if match.empty:
                    row[spec.display_name] = "N/A"
                else:
                    item = match.iloc[0]
                    row[spec.display_name] = (
                        f"{item.mean_tvd:.4f} / {item.flip_rate:.4f}"
                    )
            row["BNR"] = bnr_for(selected)
            rows.append(row)
    return pd.DataFrame(rows)


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    endpoint_root = args.endpoint_root.expanduser().resolve()
    counterfactual_root = args.counterfactual_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.primary_bag_size not in args.bag_sizes:
        raise ValueError("--primary-bag-size must be included in --bag-sizes")

    tile_manifest = pd.read_csv(endpoint_root / "tile_manifest.csv")
    variant_manifest = pd.read_csv(
        endpoint_root / "counterfactual_variant_manifest.csv"
    ).reset_index(drop=True)
    clinical = pd.read_csv(endpoint_root / "clinical_normalized.csv")
    variant_manifest, image_audit = validate_local_images(
        variant_manifest, counterfactual_root
    )
    source_patient = dict(
        zip(tile_manifest["tile_id"].astype(str), tile_manifest["patient_id"].astype(str))
    )
    variant_manifest["patient_id_resolved"] = variant_manifest[
        "source_tile_id"
    ].astype(str).map(source_patient)
    variant_manifest.to_csv(output_root / "local_variant_manifest.csv", index=False)

    pam50_folds = load_fold_map(
        endpoint_root / "models" / "resnet50" / "pam50_patient_oof_predictions.csv"
    )
    survival_folds = load_fold_map(
        endpoint_root
        / "models"
        / "resnet50"
        / "survival_patient_oof_predictions.csv"
    )

    all_summaries = []
    all_details = []
    performance_rows = []
    model_audits = []
    for model in args.models:
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported cached model: {model}")
        model_started = time.perf_counter()
        print(f"\n=== {DISPLAY_MODELS[model]} ===", flush=True)
        tile_embeddings, variant_embeddings = load_cache(
            endpoint_root, model, tile_manifest, variant_manifest
        )
        print(
            f"[{model}] cache: {tile_embeddings.shape} baseline, "
            f"{variant_embeddings.shape} counterfactual",
            flush=True,
        )

        pam_heads, pam_metrics, pam_oof = fit_tile_pam50(
            tile_manifest,
            clinical,
            tile_embeddings,
            pam50_folds,
            c_value=args.pam50_c,
            max_iter=args.pam50_max_iter,
            max_tiles_per_patient=args.max_train_tiles_per_patient,
            seed=args.seed,
        )
        joblib.dump(pam_heads, output_root / f"{model}_pam50_tile_heads.joblib")
        pam_oof.to_parquet(output_root / f"{model}_pam50_patient_oof.parquet", index=False)
        performance_rows.append(
            {
                "model_id": model,
                "model": DISPLAY_MODELS[model],
                "endpoint": "PAM50",
                "bag_size": None,
                **pam_metrics,
            }
        )
        pam_probabilities, pam_valid = predict_tile_variants(
            variant_manifest, variant_embeddings, pam_heads, pam50_folds
        )
        pam_summary, pam_detail = summarize_experiments(
            variant_manifest,
            pam_probabilities,
            pam_valid,
            model=model,
            endpoint="PAM50",
            bag_size=None,
            max_panels=args.max_panels,
        )
        all_summaries.append(pam_summary)
        all_details.append(pam_detail)
        print(
            f"[{model}] PAM50 patient macro AUC={pam_metrics['patient_macro_auc_ovr']:.4f}",
            flush=True,
        )

        survival_metrics_by_bag = {}
        for bag_size in args.bag_sizes:
            heads, survival_metrics, survival_oof = fit_fixed_bag_survival(
                tile_manifest,
                clinical,
                tile_embeddings,
                survival_folds,
                bag_size=bag_size,
                pca_components=args.survival_pca_components,
                l2_penalty=args.survival_l2,
                seed=args.seed,
            )
            joblib.dump(
                heads, output_root / f"{model}_survival_fixedbag{bag_size}_heads.joblib"
            )
            survival_oof.to_parquet(
                output_root / f"{model}_survival_fixedbag{bag_size}_oof.parquet",
                index=False,
            )
            performance_rows.append(
                {
                    "model_id": model,
                    "model": DISPLAY_MODELS[model],
                    "endpoint": "Overall survival",
                    "bag_size": bag_size,
                    "patient_macro_auc_ovr": math.nan,
                    **survival_metrics,
                }
            )
            survival_values, survival_valid = predict_fixed_bag_variants(
                variant_manifest,
                variant_embeddings,
                tile_manifest,
                tile_embeddings,
                heads,
                survival_folds,
                bag_size=bag_size,
                seed=args.seed,
            )
            survival_summary, survival_detail = summarize_experiments(
                variant_manifest,
                bernoulli_vectors(survival_values),
                survival_valid,
                model=model,
                endpoint="Overall survival",
                bag_size=bag_size,
                max_panels=args.max_panels,
            )
            all_summaries.append(survival_summary)
            all_details.append(survival_detail)
            survival_metrics_by_bag[str(bag_size)] = survival_metrics
            print(
                f"[{model}] survival M={bag_size}: "
                f"C-index={survival_metrics['c_index']:.4f}, "
                f"patients={survival_metrics['patients']}",
                flush=True,
            )
        model_audits.append(
            {
                "model_id": model,
                "feature_dim": int(tile_embeddings.shape[1]),
                "pam50": pam_metrics,
                "survival_by_bag_size": survival_metrics_by_bag,
                "elapsed_minutes": (time.perf_counter() - model_started) / 60.0,
            }
        )
        del tile_embeddings, variant_embeddings

    summaries = pd.concat(all_summaries, ignore_index=True)
    details = pd.concat(all_details, ignore_index=True)
    performances = pd.DataFrame(performance_rows)
    table = build_table(
        summaries,
        performances,
        args.primary_bag_size,
        tuple(args.models),
    )

    summaries.to_csv(output_root / "experiment_summary.csv", index=False)
    details.to_parquet(output_root / "pair_metrics.parquet", index=False)
    performances.to_csv(output_root / "performance.csv", index=False)
    table.to_csv(output_root / "table4_revised.csv", index=False)
    (output_root / "table4_revised.md").write_text(
        markdown_table(table), encoding="utf-8"
    )
    audit = {
        "protocol": {
            "pam50": "patient-disjoint patient-balanced tile-level logistic probe",
            "pam50_sensitivity_unit": "generated baseline/counterfactual tile pair",
            "survival": "patient-disjoint Cox probe on deterministic fixed-size bags",
            "survival_intervention": "one generated tile plus M-1 identical real context tiles",
            "survival_primary_bag_size": args.primary_bag_size,
            "morphology_doses": [-2, -1, 0, 1, 2],
            "seed": args.seed,
        },
        "images": image_audit,
        "baseline_tiles": len(tile_manifest),
        "baseline_patients": int(tile_manifest["patient_id"].nunique()),
        "counterfactual_variants": len(variant_manifest),
        "counterfactual_sources": int(variant_manifest["source_tile_id"].nunique()),
        "counterfactual_sources_with_baseline_tile": int(
            variant_manifest.loc[
                variant_manifest["patient_id_resolved"].notna(), "source_tile_id"
            ].nunique()
        ),
        "models": model_audits,
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
    }
    (output_root / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print("\nRevised table", flush=True)
    print(table.to_string(index=False), flush=True)
    print(f"\nWrote results to {output_root}", flush=True)


if __name__ == "__main__":
    main()


