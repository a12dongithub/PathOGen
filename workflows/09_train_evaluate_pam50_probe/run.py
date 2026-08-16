#!/usr/bin/env python3
"""Cross-fit a frozen CTransPath PAM50 probe and score counterfactual tiles."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from cpathogen.probes.ctranspath import (
    binary_metrics,
    build_encoder,
    choose_device,
    extract_embeddings,
    knob_sd,
    patient_from_stem,
    sha256,
)

TASK_NAME = "TCGA-BRCA PAM50 Basal vs Luminal A"
NEGATIVE_CLASS = "LumA"
POSITIVE_CLASS = "Basal"
LABEL_TO_INT = {NEGATIVE_CLASS: 0, POSITIVE_CLASS: 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--counterfactual-root", type=Path, required=True)
    parser.add_argument("--counterfactual-source-uri")
    parser.add_argument("--counterfactual-archive-member-prefix", default="")
    parser.add_argument("--expected-counterfactual-candidates", type=int)
    parser.add_argument("--ctranspath-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu", "mps"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c-values", type=float, nargs="+", default=(0.01, 0.1, 1.0, 10.0))
    parser.add_argument("--smoke-limit-patients", type=int)
    parser.add_argument("--smoke-limit-counterfactual", type=int)
    return parser.parse_args()


def resolve_training_paths(frame: pd.DataFrame, root: Path) -> list[Path]:
    paths = [root / value for value in frame["image_path"]]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} training images, e.g. {missing[0]}")
    return paths


def resolve_counterfactual_path(value: str, root: Path) -> Path:
    path = Path(value)
    if "images" not in path.parts:
        raise ValueError(f"Counterfactual path has no images/ component: {value}")
    return root / Path(*path.parts[path.parts.index("images") :])


def select_c_nested(
    embeddings: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    outer_fold: int,
    c_values: list[float],
    seed: int,
) -> tuple[float, list[dict[str, Any]]]:
    development = folds != outer_fold
    candidates: list[dict[str, Any]] = []
    for c_value in c_values:
        probabilities = np.full(len(labels), np.nan, dtype=np.float64)
        inner_metrics: list[dict[str, Any]] = []
        for validation_fold in sorted(set(folds[development])):
            train = development & (folds != validation_fold)
            validation = development & (folds == validation_fold)
            classifier = LogisticRegression(
                C=c_value, class_weight="balanced", max_iter=2_000, random_state=seed
            )
            classifier.fit(embeddings[train], labels[train])
            probability = classifier.predict_proba(embeddings[validation])[:, 1]
            probabilities[validation] = probability
            inner_metrics.append(
                {"validation_fold": int(validation_fold), **binary_metrics(labels[validation], probability)}
            )
        metrics = binary_metrics(labels[development], probabilities[development])
        candidates.append({"C": float(c_value), "pooled_inner_validation": metrics, "inner_folds": inner_metrics})
    selected = max(
        candidates,
        key=lambda row: (
            row["pooled_inner_validation"]["roc_auc"],
            -row["pooled_inner_validation"]["log_loss"],
        ),
    )
    return float(selected["C"]), candidates


def fit_cross_fitted_heads(
    embeddings: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    c_values: list[float],
    seed: int,
) -> tuple[dict[int, LogisticRegression], np.ndarray, dict[str, Any]]:
    fold_ids = sorted(int(value) for value in np.unique(folds))
    heads: dict[int, LogisticRegression] = {}
    oof = np.full(len(labels), np.nan, dtype=np.float64)
    fold_results: list[dict[str, Any]] = []
    for outer_fold in fold_ids:
        selected_c, selection = select_c_nested(
            embeddings, labels, folds, outer_fold, c_values, seed
        )
        train = folds != outer_fold
        test = folds == outer_fold
        classifier = LogisticRegression(
            C=selected_c, class_weight="balanced", max_iter=2_000, random_state=seed
        )
        classifier.fit(embeddings[train], labels[train])
        probability = classifier.predict_proba(embeddings[test])[:, 1]
        oof[test] = probability
        heads[outer_fold] = classifier
        fold_results.append(
            {
                "outer_fold": outer_fold,
                "selected_C": selected_c,
                "train_patients": int(train.sum()),
                "test_patients": int(test.sum()),
                "test": binary_metrics(labels[test], probability),
                "selection": selection,
            }
        )
    if np.isnan(oof).any():
        raise ValueError("Some patients did not receive an out-of-fold prediction")
    return heads, oof, {"overall_oof": binary_metrics(labels, oof), "folds": fold_results}


def score_counterfactuals(
    embeddings: np.ndarray,
    source_patients: pd.Series,
    patient_folds: dict[str, int],
    heads: dict[int, LogisticRegression],
) -> tuple[np.ndarray, list[str], list[str]]:
    per_fold = {fold: head.predict_proba(embeddings)[:, 1] for fold, head in heads.items()}
    probabilities = np.empty(len(embeddings), dtype=np.float64)
    methods: list[str] = []
    scoring_folds: list[str] = []
    all_folds = sorted(heads)
    for index, patient in enumerate(source_patients):
        if patient in patient_folds:
            fold = patient_folds[patient]
            probabilities[index] = per_fold[fold][index]
            methods.append("source_patient_held_out_fold")
            scoring_folds.append(str(fold))
        else:
            probabilities[index] = float(np.mean([per_fold[fold][index] for fold in all_folds]))
            methods.append("all_fold_ensemble_source_not_in_binary_cohort")
            scoring_folds.append("|".join(map(str, all_folds)))
    return probabilities, methods, scoring_folds


def validate_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    training_manifest = args.training_root / "tiles.csv"
    counterfactual_manifest = args.counterfactual_root / "images.csv"
    for path in (training_manifest, counterfactual_manifest, args.ctranspath_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {args.output_dir}")
    return training_manifest, counterfactual_manifest


def main() -> None:
    args = parse_args()
    training_manifest, counterfactual_manifest = validate_inputs(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    training = pd.read_csv(training_manifest)
    required = {"tile_id", "image_path", "patient_id", "label", "outer_fold"}
    if not required.issubset(training.columns):
        raise ValueError(f"Training manifest lacks {sorted(required - set(training.columns))}")
    if set(training["label"]) != set(LABEL_TO_INT):
        raise ValueError(f"Expected labels {sorted(LABEL_TO_INT)}, found {sorted(training['label'].unique())}")
    if args.smoke_limit_patients:
        selected = (
            training[["patient_id", "label", "outer_fold"]]
            .drop_duplicates()
            .groupby(["outer_fold", "label"], group_keys=False)
            .head(args.smoke_limit_patients)
        )
        training = training[training["patient_id"].isin(selected["patient_id"])].copy()

    counterfactuals = pd.read_csv(counterfactual_manifest)
    if args.smoke_limit_counterfactual:
        counterfactuals = counterfactuals.head(args.smoke_limit_counterfactual).copy()
    counterfactuals["source_patient_id"] = counterfactuals["stem"].map(patient_from_stem)

    encoder = build_encoder(args.ctranspath_checkpoint, device)
    real_paths = resolve_training_paths(training, args.training_root)
    real_tile_embeddings = extract_embeddings(
        encoder,
        real_paths,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        description="real TCGA-BRCA tiles",
    )
    labels = training["label"].map(LABEL_TO_INT).to_numpy(dtype=np.int64)
    folds = training["outer_fold"].to_numpy(dtype=np.int64)
    if len(set(folds)) != 5:
        raise ValueError(f"Expected five outer folds, found {sorted(set(folds))}")
    heads, oof, metrics = fit_cross_fitted_heads(
        real_tile_embeddings, labels, folds, list(args.c_values), args.seed
    )
    tile_oof = training.copy()
    tile_oof["label_index"] = labels
    tile_oof["probability_Basal_oof"] = oof
    tile_oof["probability_LumA_oof"] = 1.0 - oof
    tile_oof["predicted_label_oof"] = np.where(oof >= 0.5, POSITIVE_CLASS, NEGATIVE_CLASS)
    tile_oof.to_csv(args.output_dir / "tile_oof_predictions.csv", index=False)
    patients = (
        tile_oof.groupby(["patient_id", "label", "outer_fold"], as_index=False)
        .agg(
            tile_count=("tile_id", "size"),
            probability_Basal_oof=("probability_Basal_oof", "mean"),
        )
        .sort_values(["outer_fold", "label", "patient_id"], kind="stable")
    )
    patients["probability_LumA_oof"] = 1.0 - patients["probability_Basal_oof"]
    patients["predicted_label_oof"] = np.where(
        patients["probability_Basal_oof"] >= 0.5, POSITIVE_CLASS, NEGATIVE_CLASS
    )
    patients.to_csv(args.output_dir / "patient_oof_predictions.csv", index=False)
    patient_labels_int = patients["label"].map(LABEL_TO_INT).to_numpy(dtype=np.int64)
    metrics["overall_oof_tile"] = metrics.pop("overall_oof")
    metrics["overall_oof_patient"] = binary_metrics(
        patient_labels_int, patients["probability_Basal_oof"].to_numpy()
    )
    np.savez_compressed(
        args.output_dir / "real_embeddings.npz",
        tile_embeddings=real_tile_embeddings,
        tile_ids=training["tile_id"].to_numpy(),
        labels=labels,
        outer_folds=folds,
    )
    joblib.dump(
        {
            "heads_by_held_out_fold": heads,
            "positive_class": POSITIVE_CLASS,
            "negative_class": NEGATIVE_CLASS,
            "task": TASK_NAME,
            "training_unit": "tile with patient-inherited PAM50 label",
        },
        args.output_dir / "cross_fitted_classifiers.joblib",
    )
    for fold, head in heads.items():
        np.savez(
            args.output_dir / f"head_fold_{fold}.npz",
            coef=head.coef_, intercept=head.intercept_, classes=head.classes_
        )

    cf_paths = [resolve_counterfactual_path(value, args.counterfactual_root) for value in counterfactuals["image_path"]]
    missing = [path for path in cf_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} counterfactual images, e.g. {missing[0]}")
    cf_embeddings = extract_embeddings(
        encoder,
        cf_paths,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        description="counterfactual tiles",
    )
    patient_folds = dict(zip(patients["patient_id"], patients["outer_fold"], strict=True))
    patient_labels = dict(zip(patients["patient_id"], patients["label"], strict=True))
    probabilities, methods, scoring_folds = score_counterfactuals(
        cf_embeddings, counterfactuals["source_patient_id"], patient_folds, heads
    )
    results = counterfactuals.copy()
    results["model_id"] = "ctranspath_frozen_pam50_crossfit_l2_logistic"
    results["task"] = TASK_NAME
    results["source_patient_id"] = counterfactuals["source_patient_id"]
    results["source_pam50_label"] = results["source_patient_id"].map(patient_labels)
    results["scoring_method"] = methods
    results["scoring_folds"] = scoring_folds
    results["knob_sd"] = [
        knob_sd(parameters, condition)
        for parameters, condition in zip(results["intervention_parameters"], results["condition"], strict=True)
    ]
    results["probability_Basal"] = probabilities
    results["probability_LumA"] = 1.0 - probabilities
    results["predicted_class_index"] = (probabilities >= 0.5).astype(np.int64)
    results["predicted_label"] = np.where(probabilities >= 0.5, POSITIVE_CLASS, NEGATIVE_CLASS)
    results["relative_image_path"] = [str(path.relative_to(args.counterfactual_root)) for path in cf_paths]
    if args.counterfactual_source_uri:
        source = args.counterfactual_source_uri.rstrip("/")
        results["counterfactual_source_uri"] = source
        if source.endswith(".zip"):
            prefix = Path(args.counterfactual_archive_member_prefix)
            results["counterfactual_archive_member"] = [str(prefix / value) for value in results["relative_image_path"]]
            results["counterfactual_gcs_uri"] = pd.NA
        else:
            results["counterfactual_archive_member"] = pd.NA
            results["counterfactual_gcs_uri"] = [f"{source}/{value}" for value in results["relative_image_path"]]
    else:
        results["counterfactual_source_uri"] = pd.NA
        results["counterfactual_archive_member"] = pd.NA
        results["counterfactual_gcs_uri"] = pd.NA

    if not args.smoke_limit_counterfactual:
        candidate_count = int(results["candidate_id"].nunique())
        if args.expected_counterfactual_candidates and candidate_count != args.expected_counterfactual_candidates:
            raise ValueError(f"Expected {args.expected_counterfactual_candidates} candidates, found {candidate_count}")
        counts = results.groupby("candidate_id")["condition"].nunique()
        if len(results) != candidate_count * 4 or not (counts == 4).all():
            raise ValueError("Every candidate must have exactly four conditions")
        if set(results["knob_sd"]) != {0.0, 0.5, 1.0, 1.5}:
            raise ValueError("Unexpected counterfactual SD levels")

    baseline = (
        results.loc[results["condition"] == "baseline", ["candidate_id", "probability_Basal"]]
        .drop_duplicates("candidate_id")
        .rename(columns={"probability_Basal": "baseline_probability_Basal"})
    )
    results = results.merge(baseline, on="candidate_id", how="left", validate="many_to_one")
    results["delta_probability_Basal"] = results["probability_Basal"] - results["baseline_probability_Basal"]
    baseline_classes = (
        results.loc[results["condition"] == "baseline", ["candidate_id", "predicted_label"]]
        .drop_duplicates("candidate_id")
        .rename(columns={"predicted_label": "baseline_predicted_label"})
    )
    results = results.merge(
        baseline_classes, on="candidate_id", how="left", validate="many_to_one"
    )
    results["class_flipped_from_baseline"] = (
        results["predicted_label"] != results["baseline_predicted_label"]
    )
    results.to_csv(args.output_dir / "counterfactual_predictions.csv", index=False)
    results.to_parquet(args.output_dir / "counterfactual_predictions.parquet", index=False)
    summary = (
        results.groupby(["condition", "knob_sd"], as_index=False, dropna=False)
        .agg(
            tile_count=("candidate_id", "size"),
            mean_probability_Basal=("probability_Basal", "mean"),
            std_probability_Basal=("probability_Basal", "std"),
            mean_delta_probability_Basal=("delta_probability_Basal", "mean"),
            class_flip_rate=("class_flipped_from_baseline", "mean"),
        )
        .sort_values("knob_sd")
    )
    summary.to_csv(args.output_dir / "counterfactual_summary.csv", index=False)
    labeled_summary = (
        results.dropna(subset=["source_pam50_label"])
        .groupby(["source_pam50_label", "condition", "knob_sd"], as_index=False)
        .agg(
            tile_count=("candidate_id", "size"),
            patient_count=("source_patient_id", "nunique"),
            mean_probability_Basal=("probability_Basal", "mean"),
            mean_delta_probability_Basal=("delta_probability_Basal", "mean"),
        )
        .sort_values(["source_pam50_label", "knob_sd"])
    )
    labeled_summary.to_csv(args.output_dir / "counterfactual_summary_by_true_pam50.csv", index=False)

    completed_at = datetime.now(timezone.utc).isoformat()
    metrics.update(
        {
            "task": TASK_NAME,
            "positive_class": POSITIVE_CLASS,
            "training_unit": "tile with patient-inherited PAM50 label",
            "folding_unit": "patient",
            "patient_aggregation": "mean tile probability",
            "encoder": "CTransPath",
            "encoder_checkpoint_sha256": sha256(args.ctranspath_checkpoint),
            "device": str(device),
            "real_tiles": len(training),
            "real_patients": len(patients),
            "counterfactual_rows": len(results),
            "counterfactual_candidates": int(results["candidate_id"].nunique()),
            "counterfactual_source_patients": int(results["source_patient_id"].nunique()),
            "labeled_counterfactual_candidates": int(results.loc[results["source_pam50_label"].notna(), "candidate_id"].nunique()),
            "scoring_method_counts": results[["candidate_id", "scoring_method"]].drop_duplicates()["scoring_method"].value_counts().to_dict(),
            "completed_at": completed_at,
        }
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "created_at": completed_at,
        "task": TASK_NAME,
        "encoder": "CTransPath",
        "encoder_frozen": True,
        "head": "five tile-trained, patient-disjoint, class-balanced L2 logistic regressions",
        "counterfactual_source_uri": args.counterfactual_source_uri,
        "counterfactual_archive_member_prefix": args.counterfactual_archive_member_prefix or None,
        "training_manifest_sha256": sha256(training_manifest),
        "counterfactual_manifest_sha256": sha256(counterfactual_manifest),
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(args.output_dir.iterdir())
            if path.is_file() and path.name != "run_manifest.json"
        },
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(results)} counterfactual predictions to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
