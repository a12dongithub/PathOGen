#!/usr/bin/env python3
"""Score counterfactuals by replacing their source tile in each patient bag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from cpathogen.endpoints.clinical import PAM50_CLASSES, load_clinical_matrix
from cpathogen.endpoints.encoders import (
    build_encoder,
    choose_device,
    extract_embeddings_sharded,
    release_encoder,
)
from cpathogen.endpoints.jsonio import write_jsonl
from cpathogen.endpoints.paper_xai import filter_paper_variants
from cpathogen.endpoints.probes import assign_pam50_folds, assign_survival_folds
from cpathogen.endpoints.variants import (
    discover_variant_manifests,
    normalize_variant_manifests,
)

HORIZONS = np.asarray((1826, 3652), dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical-tsi", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant-root", type=Path)
    parser.add_argument("--variant-manifests", type=Path, nargs="*")
    parser.add_argument(
        "--encoders",
        nargs="+",
        default=("uni2h", "ctranspath", "virchow2", "resnet50"),
    )
    parser.add_argument("--ctranspath-checkpoint", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--paper-five-only",
        action="store_true",
        help="Embed only the reference/target images used by the five paper columns.",
    )
    return parser.parse_args()


def _variant_features(
    encoder_name: str,
    variants: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    cache = args.output_root / "embedding_cache" / f"{encoder_name}_counterfactuals.npz"
    variant_ids = variants["variant_id"].to_numpy(dtype=str)
    if cache.is_file():
        payload = np.load(cache, allow_pickle=False)
        if np.array_equal(payload["variant_ids"].astype(str), variant_ids):
            print(f"[{encoder_name}] reusing counterfactual embeddings", flush=True)
            return payload["embeddings"].astype(np.float32, copy=False)
    bundle = build_encoder(
        encoder_name,
        device=device,
        ctranspath_checkpoint=args.ctranspath_checkpoint,
    )
    try:
        embeddings = extract_embeddings_sharded(
            bundle,
            [Path(value) for value in variants["image_path"]],
            variant_ids,
            shard_root=args.output_root / "embedding_cache" / "shards",
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            description=f"{encoder_name} counterfactuals",
        )
    finally:
        release_encoder(bundle)
    temporary = cache.with_suffix(".npz.part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, variant_ids=variant_ids, embeddings=embeddings)
    temporary.replace(cache)
    return embeddings


def _replacement_features(
    variants: pd.DataFrame,
    variant_embeddings: np.ndarray,
    manifest: pd.DataFrame,
    real_embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    groups = manifest.groupby("patient_id", sort=False).indices
    sums = {
        patient: real_embeddings[np.asarray(indices)].sum(axis=0)
        for patient, indices in groups.items()
    }
    counts = {patient: len(indices) for patient, indices in groups.items()}
    source_indices = {
        str(row.tile_id): index
        for index, row in manifest.reset_index(drop=True).iterrows()
        if int(getattr(row, "augmentation_code", 0)) == 0
    }
    features = np.zeros_like(variant_embeddings)
    baseline = np.zeros_like(variant_embeddings)
    status: list[str] = []
    for index, row in variants.iterrows():
        patient = row["patient_id"]
        source = row["source_tile_id"]
        if patient not in sums:
            status.append("patient_not_in_training_bags")
            continue
        if source not in source_indices:
            status.append("source_tile_not_in_selected_training_bag")
            continue
        source_index = source_indices[source]
        baseline_mean = sums[patient] / counts[patient]
        replacement_mean = (
            sums[patient] - real_embeddings[source_index] + variant_embeddings[index]
        ) / counts[patient]
        baseline_norm = np.linalg.norm(baseline_mean)
        replacement_norm = np.linalg.norm(replacement_mean)
        baseline[index] = (
            baseline_mean / baseline_norm if baseline_norm else baseline_mean
        )
        features[index] = (
            replacement_mean / replacement_norm
            if replacement_norm
            else replacement_mean
        )
        status.append("ok")
    return features, baseline, status


def _score_pam50(
    records: list[dict],
    variants: pd.DataFrame,
    replacement: np.ndarray,
    baseline: np.ndarray,
    valid: np.ndarray,
    patient_frame: pd.DataFrame,
    heads: dict,
    args: argparse.Namespace,
    model_name: str,
) -> None:
    cohort = patient_frame[patient_frame["pam50"].isin(PAM50_CLASSES)].reset_index(
        drop=True
    )
    folds = assign_pam50_folds(cohort, n_folds=args.folds, seed=args.seed)
    fold_map = dict(zip(cohort["patient_id"], folds, strict=True))
    for index, row in variants.iterrows():
        patient = row["patient_id"]
        status = "ok" if valid[index] else "bag_replacement_unavailable"
        prediction = None
        baseline_prediction = None
        fold = fold_map.get(patient)
        if status == "ok" and fold is None:
            status = "pam50_label_missing"
        if status == "ok":
            head = heads[int(fold)]
            probability = head.predict_proba(replacement[index : index + 1])[0]
            base_probability = head.predict_proba(baseline[index : index + 1])[0]
            prediction = {
                "probabilities": {
                    name: float(probability[i]) for i, name in enumerate(PAM50_CLASSES)
                },
                "predicted_class": PAM50_CLASSES[int(probability.argmax())],
            }
            baseline_prediction = {
                "probabilities": {
                    name: float(base_probability[i])
                    for i, name in enumerate(PAM50_CLASSES)
                },
                "predicted_class": PAM50_CLASSES[int(base_probability.argmax())],
            }
        records.append(
            {
                **row.to_dict(),
                "model_id": model_name,
                "endpoint": "PAM50_four_class",
                "scoring_fold": None if fold is None else int(fold),
                "status": status,
                "prediction": prediction,
                "baseline_prediction": baseline_prediction,
            }
        )


def _score_survival(
    records: list[dict],
    variants: pd.DataFrame,
    replacement: np.ndarray,
    baseline: np.ndarray,
    valid: np.ndarray,
    patient_frame: pd.DataFrame,
    heads: dict,
    args: argparse.Namespace,
    model_name: str,
) -> None:
    cohort = patient_frame[
        patient_frame[["survival_time_days", "survival_event"]].notna().all(axis=1)
    ].reset_index(drop=True)
    folds = assign_survival_folds(cohort, n_folds=args.folds, seed=args.seed)
    fold_map = dict(zip(cohort["patient_id"], folds, strict=True))
    for index, row in variants.iterrows():
        patient = row["patient_id"]
        status = "ok" if valid[index] else "bag_replacement_unavailable"
        prediction = None
        baseline_prediction = None
        fold = fold_map.get(patient)
        if status == "ok" and fold is None:
            status = "survival_label_missing"
        if status == "ok":
            head = heads[int(fold)]
            risk = float(head.predict_risk(replacement[index : index + 1])[0])
            base_risk = float(head.predict_risk(baseline[index : index + 1])[0])
            survival = head.predict_survival(replacement[index : index + 1], HORIZONS)[
                0
            ]
            base_survival = head.predict_survival(
                baseline[index : index + 1], HORIZONS
            )[0]
            prediction = {
                "log_risk": risk,
                "survival_probability_5y": float(survival[0]),
                "survival_probability_10y": float(survival[1]),
            }
            baseline_prediction = {
                "log_risk": base_risk,
                "survival_probability_5y": float(base_survival[0]),
                "survival_probability_10y": float(base_survival[1]),
            }
        records.append(
            {
                **row.to_dict(),
                "model_id": model_name,
                "endpoint": "overall_survival",
                "scoring_fold": None if fold is None else int(fold),
                "status": status,
                "prediction": prediction,
                "baseline_prediction": baseline_prediction,
            }
        )


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    paths = list(args.variant_manifests or [])
    if args.variant_root:
        paths.extend(discover_variant_manifests(args.variant_root))
    if not paths:
        raise SystemExit("Supply --variant-root or --variant-manifests")
    variants = normalize_variant_manifests(paths)
    if args.paper_five_only:
        variants = filter_paper_variants(variants)
    variants.to_csv(
        args.output_root / "counterfactual_variant_manifest.csv", index=False
    )
    manifest = pd.read_csv(args.output_root / "tile_manifest.csv")
    clinical = load_clinical_matrix(args.clinical_tsi)
    patient_frame = (
        manifest.groupby("patient_id", as_index=False)
        .agg(tile_count=("tile_id", "size"))
        .merge(clinical, on="patient_id", how="left", validate="one_to_one")
        .sort_values("patient_id", kind="stable")
        .reset_index(drop=True)
    )
    device = choose_device(args.device)
    for model_name in args.encoders:
        model_dir = args.output_root / "models" / model_name
        cache = np.load(
            args.output_root / "embedding_cache" / f"{model_name}_tiles.npz",
            allow_pickle=False,
        )
        if not np.array_equal(
            cache["tile_ids"].astype(str), manifest["tile_id"].to_numpy(dtype=str)
        ):
            raise ValueError(
                f"{model_name} real embedding cache does not match tile_manifest.csv"
            )
        real_embeddings = cache["embeddings"].astype(np.float32, copy=False)
        variant_embeddings = _variant_features(model_name, variants, args, device)
        replacement, baseline, replacement_status = _replacement_features(
            variants, variant_embeddings, manifest, real_embeddings
        )
        valid = np.asarray([value == "ok" for value in replacement_status])
        records: list[dict] = []
        _score_pam50(
            records,
            variants,
            replacement,
            baseline,
            valid,
            patient_frame,
            joblib.load(model_dir / "pam50_crossfit_heads.joblib"),
            args,
            model_name,
        )
        _score_survival(
            records,
            variants,
            replacement,
            baseline,
            valid,
            patient_frame,
            joblib.load(model_dir / "survival_crossfit_heads.joblib"),
            args,
            model_name,
        )
        for record in records:
            record["schema_version"] = 1
        write_jsonl(model_dir / "counterfactual_predictions.jsonl", records)
        flat = pd.DataFrame(
            {
                **{
                    column: pd.concat([variants, variants], ignore_index=True)[column]
                    for column in variants
                },
                "endpoint": [record["endpoint"] for record in records],
                "status": [record["status"] for record in records],
                "scoring_fold": [record["scoring_fold"] for record in records],
                "prediction_json": [
                    json.dumps(record["prediction"], sort_keys=True)
                    for record in records
                ],
                "baseline_prediction_json": [
                    json.dumps(record["baseline_prediction"], sort_keys=True)
                    for record in records
                ],
            }
        )
        flat.to_parquet(model_dir / "counterfactual_predictions.parquet", index=False)
        print(
            f"[{model_name}] wrote {len(records)} endpoint-variant JSON records",
            flush=True,
        )


if __name__ == "__main__":
    main()
