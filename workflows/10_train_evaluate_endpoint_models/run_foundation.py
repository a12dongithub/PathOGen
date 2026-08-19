#!/usr/bin/env python3
"""Train and evaluate patient-level PAM50 and survival heads on frozen encoders."""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from cpathogen.endpoints.clinical import (
    PAM50_CLASSES,
    build_tile_manifest,
    load_clinical_matrix,
)
from cpathogen.endpoints.encoders import (
    build_encoder,
    choose_device,
    extract_embeddings_sharded,
    release_encoder,
)
from cpathogen.endpoints.jsonio import write_json
from cpathogen.endpoints.probes import (
    aggregate_patient_embeddings,
    assign_pam50_folds,
    assign_survival_folds,
    fit_pam50_crossfit,
    fit_survival_crossfit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical-tsi", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--encoders",
        nargs="+",
        default=("uni2h", "ctranspath", "virchow2", "resnet50"),
    )
    parser.add_argument("--ctranspath-checkpoint", type=Path)
    parser.add_argument("--max-tiles-per-patient", type=int, default=256)
    parser.add_argument("--minimum-views-per-patient", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--pam50-c", type=float, default=1.0)
    parser.add_argument("--survival-pca-components", type=int, default=64)
    parser.add_argument("--survival-l2", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--smoke-patients", type=int)
    parser.add_argument("--overwrite-cache", action="store_true")
    return parser.parse_args()


def _resolve_images_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "images").is_dir():
        return path / "images"
    return path


def _augment_manifest(manifest: pd.DataFrame, minimum_views: int) -> pd.DataFrame:
    """Add deterministic flip/rotation views only for patients with very few tiles."""
    rows: list[dict[str, Any]] = []
    for _, group in manifest.groupby("patient_id", sort=True):
        records = group.to_dict("records")
        for record in records:
            rows.append(
                {**record, "augmentation_code": 0, "source_tile_id": record["tile_id"]}
            )
        target = max(len(records), minimum_views)
        extra = target - len(records)
        for index in range(extra):
            source = records[index % len(records)]
            code = 1 + (index // len(records)) % 3
            rows.append(
                {
                    **source,
                    "tile_id": f"{source['tile_id']}__aug{code}_{index}",
                    "augmentation_code": code,
                    "source_tile_id": source["tile_id"],
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["patient_id", "source_tile_id", "augmentation_code"], kind="stable"
        )
        .reset_index(drop=True)
    )


def _smoke_subset(
    clinical: pd.DataFrame, manifest: pd.DataFrame, count: int | None, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not count:
        return clinical, manifest
    rng = np.random.default_rng(seed)
    candidates = sorted(set(manifest["patient_id"]) & set(clinical["patient_id"]))
    pam = clinical.set_index("patient_id").loc[candidates, "pam50"]
    selected: list[str] = []
    for label in PAM50_CLASSES:
        group = pam.index[pam == label].to_numpy()
        if len(group):
            selected.extend(
                rng.choice(group, size=min(count, len(group)), replace=False).tolist()
            )
    survival_only = [patient for patient in candidates if patient not in selected]
    rng.shuffle(survival_only)
    selected.extend(survival_only[: max(0, count * 4 - len(selected))])
    selected_set = set(selected)
    return (
        clinical[clinical["patient_id"].isin(selected_set)].reset_index(drop=True),
        manifest[manifest["patient_id"].isin(selected_set)].reset_index(drop=True),
    )


def _load_or_extract(
    encoder_name: str,
    manifest: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    cache_dir = args.output_root / "embedding_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{encoder_name}_tiles.npz"
    tile_ids = manifest["tile_id"].to_numpy(dtype=str)
    if cache.is_file() and not args.overwrite_cache:
        payload = np.load(cache, allow_pickle=False)
        if np.array_equal(payload["tile_ids"].astype(str), tile_ids):
            print(f"[{encoder_name}] reusing {cache}", flush=True)
            return payload["embeddings"].astype(np.float32, copy=False)
        print(f"[{encoder_name}] cache manifest mismatch; recomputing", flush=True)

    bundle = build_encoder(
        encoder_name,
        device=device,
        ctranspath_checkpoint=args.ctranspath_checkpoint,
    )
    try:
        embeddings = extract_embeddings_sharded(
            bundle,
            [Path(value) for value in manifest["image_path"]],
            tile_ids,
            shard_root=cache_dir / "shards",
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            description=f"{encoder_name} tiles",
            augmentation_codes=manifest["augmentation_code"].astype(int).tolist(),
        )
        temporary = cache.with_suffix(".npz.part")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, tile_ids=tile_ids, embeddings=embeddings)
        temporary.replace(cache)
        metadata = {
            "encoder": bundle.name,
            "model_id": bundle.model_id,
            "feature_dim": bundle.feature_dim,
            "tiles_or_views": len(manifest),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(cache.with_suffix(".json"), metadata)
        return embeddings
    finally:
        release_encoder(bundle)


def evaluate_encoder(
    encoder_name: str,
    patient_frame: pd.DataFrame,
    patient_features: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    output = args.output_root / "models" / encoder_name
    output.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []

    pam_mask = patient_frame["pam50"].isin(PAM50_CLASSES).to_numpy()
    pam = patient_frame.loc[pam_mask].reset_index(drop=True)
    pam_features = patient_features[pam_mask]
    pam_labels = (
        pam["pam50"].map({name: i for i, name in enumerate(PAM50_CLASSES)}).to_numpy()
    )
    pam_folds = assign_pam50_folds(pam, n_folds=args.folds, seed=args.seed)
    pam_heads, pam_probabilities, pam_metrics = fit_pam50_crossfit(
        pam_features,
        pam_labels,
        pam_folds,
        seed=args.seed,
        c_value=args.pam50_c,
    )
    pam_predictions = pam[["patient_id", "tile_count", "pam50"]].copy()
    pam_predictions["outer_fold"] = pam_folds
    pam_predictions["true_class_index"] = pam_labels
    pam_predictions["predicted_class_index"] = pam_probabilities.argmax(axis=1)
    pam_predictions["predicted_class"] = [
        PAM50_CLASSES[index] for index in pam_probabilities.argmax(axis=1)
    ]
    for index, name in enumerate(PAM50_CLASSES):
        pam_predictions[f"probability_{name}"] = pam_probabilities[:, index]
    pam_predictions.to_csv(output / "pam50_patient_oof_predictions.csv", index=False)
    pam_predictions.to_parquet(
        output / "pam50_patient_oof_predictions.parquet", index=False
    )
    pam_metrics.update(
        {
            "schema_version": 1,
            "model": encoder_name,
            "endpoint": "PAM50_four_class",
            "unit": "patient",
            "classes": list(PAM50_CLASSES),
            "split": f"{args.folds}-fold patient-disjoint OOF",
            "head": "class-balanced L2 multinomial logistic regression",
            "patients": len(pam),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_json(output / "pam50_metrics.json", pam_metrics)
    joblib.dump(pam_heads, output / "pam50_crossfit_heads.joblib")
    summaries.append(
        {
            "model": encoder_name,
            "endpoint": "PAM50",
            "patients": len(pam),
            "macro_auc": pam_metrics["overall_oof"]["macro_roc_auc_ovr"],
            "macro_f1": pam_metrics["overall_oof"]["macro_f1"],
            "accuracy": pam_metrics["overall_oof"]["overall_accuracy"],
            "c_index": np.nan,
        }
    )

    survival_mask = (
        patient_frame[["survival_time_days", "survival_event"]]
        .notna()
        .all(axis=1)
        .to_numpy()
    )
    survival = patient_frame.loc[survival_mask].reset_index(drop=True)
    survival_features = patient_features[survival_mask]
    survival_folds = assign_survival_folds(survival, n_folds=args.folds, seed=args.seed)
    times = survival["survival_time_days"].to_numpy(dtype=float)
    events = survival["survival_event"].to_numpy(dtype=int)
    survival_heads, survival_predictions, survival_metrics = fit_survival_crossfit(
        survival_features,
        times,
        events,
        survival_folds,
        seed=args.seed,
        pca_components=args.survival_pca_components,
        l2_penalty=args.survival_l2,
    )
    survival_output = survival[
        ["patient_id", "tile_count", "survival_time_days", "survival_event"]
    ].copy()
    survival_output["outer_fold"] = survival_folds
    survival_output = pd.concat([survival_output, survival_predictions], axis=1)
    survival_output.to_csv(output / "survival_patient_oof_predictions.csv", index=False)
    survival_output.to_parquet(
        output / "survival_patient_oof_predictions.parquet", index=False
    )
    survival_metrics.update(
        {
            "schema_version": 1,
            "model": encoder_name,
            "endpoint": "overall_survival",
            "unit": "patient",
            "duration": "overall_survival days",
            "event": "status (1=event, 0=censored)",
            "split": f"{args.folds}-fold patient-disjoint OOF",
            "head": "L2-regularized Cox proportional hazards after train-fitted PCA",
            "patients": len(survival),
            "events": int(events.sum()),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_json(output / "survival_metrics.json", survival_metrics)
    joblib.dump(survival_heads, output / "survival_crossfit_heads.joblib")
    summaries.append(
        {
            "model": encoder_name,
            "endpoint": "overall_survival",
            "patients": len(survival),
            "macro_auc": np.nan,
            "macro_f1": np.nan,
            "accuracy": np.nan,
            "c_index": survival_metrics["overall_oof_c_index"],
        }
    )
    return summaries


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    images_dir = _resolve_images_dir(args.images_dir)
    clinical = load_clinical_matrix(args.clinical_tsi)
    manifest = build_tile_manifest(
        images_dir,
        clinical,
        max_tiles_per_patient=args.max_tiles_per_patient,
        seed=args.seed,
    )
    clinical, manifest = _smoke_subset(
        clinical, manifest, args.smoke_patients, args.seed
    )
    manifest = _augment_manifest(manifest, args.minimum_views_per_patient)
    manifest.to_csv(args.output_root / "tile_manifest.csv", index=False)
    manifest.to_parquet(args.output_root / "tile_manifest.parquet", index=False)
    clinical.to_csv(args.output_root / "clinical_normalized.csv", index=False)

    summaries: list[dict[str, Any]] = []
    for encoder_name in args.encoders:
        print(f"\n=== {encoder_name} on {device} ===", flush=True)
        tile_embeddings = _load_or_extract(encoder_name, manifest, args, device)
        patient_index, patient_embeddings = aggregate_patient_embeddings(
            manifest, tile_embeddings
        )
        patient_frame = patient_index.merge(
            clinical,
            on="patient_id",
            how="left",
            validate="one_to_one",
        )
        summaries.extend(
            evaluate_encoder(encoder_name, patient_frame, patient_embeddings, args)
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_root / "foundation_performance.csv", index=False)
    write_json(
        args.output_root / "foundation_performance.json",
        {
            "schema_version": 1,
            "rows": summary.to_dict("records"),
            "protocol": {
                "unit": "patient",
                "tile_pooling": "mean of L2-normalized tile embeddings, then L2 normalize",
                "folds": args.folds,
                "max_tiles_per_patient": args.max_tiles_per_patient,
                "minimum_views_per_patient": args.minimum_views_per_patient,
                "augmentation": "deterministic flips/180-degree rotation only below minimum views",
                "class_balancing": "inverse-frequency class weights in PAM50 head",
                "seed": args.seed,
            },
            "recurrence": {
                "evaluable": False,
                "reason": "Clinical.tsi contains no recurrence time/event or MammaPrint labels.",
            },
        },
    )
    print(f"Wrote endpoint results to {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
