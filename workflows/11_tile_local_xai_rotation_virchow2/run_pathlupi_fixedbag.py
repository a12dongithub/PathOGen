#!/usr/bin/env python3
"""Evaluate released PathLUPI BRCA survival folds with fixed-size tile bags.

Each counterfactual replaces one tile in an otherwise identical deterministic
bag. This removes the variable dilution introduced by averaging a modified tile
into every available tile for a patient. The script consumes cached CONCH
embeddings and never re-encodes images.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from run_local_xai_rerun import (
    DISPLAY_MODELS,
    EXPERIMENTS,
    bernoulli_vectors,
    bnr_for,
    concordance_index,
    markdown_table,
    stable_rng,
    summarize_experiments,
)

MODEL_ID = "pathlupi_conch"
DISPLAY_NAME = DISPLAY_MODELS[MODEL_ID]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-root", type=Path, required=True)
    parser.add_argument("--pathlupi-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--base-results-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bag-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-sources", type=int)
    return parser.parse_args()


def load_conch_cache(
    endpoint_root: Path,
    tile_manifest: pd.DataFrame,
    variant_manifest: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    cache_root = endpoint_root / "embedding_cache"
    tile_cache = np.load(cache_root / "conch_pretrained_bags.npz", allow_pickle=False)
    variant_cache = np.load(
        cache_root / "conch_counterfactuals_pretrained.npz", allow_pickle=False
    )
    expected_tiles = tile_manifest["tile_id"].astype(str).to_numpy()
    expected_variants = variant_manifest["variant_id"].astype(str).to_numpy()
    if not np.array_equal(tile_cache["tile_ids"].astype(str), expected_tiles):
        raise ValueError("CONCH baseline cache does not match tile_manifest.csv")
    if not np.array_equal(
        variant_cache["tile_ids"].astype(str), expected_variants
    ):
        raise ValueError("CONCH counterfactual cache does not match the variant manifest")
    tiles = tile_cache["embeddings"].astype(np.float32, copy=False)
    variants = variant_cache["embeddings"].astype(np.float32, copy=False)
    if not np.isfinite(tiles).all() or not np.isfinite(variants).all():
        raise ValueError("CONCH cache contains non-finite values")
    return tiles, variants


def official_fold_map(pathlupi_root: Path) -> dict[str, int]:
    frame = pd.read_csv(pathlupi_root / "splits" / "survival" / "BRCA_Splits.csv")
    result: dict[str, int] = {}
    for _, row in frame.iterrows():
        validation = [fold for fold in range(5) if row[f"Fold {fold}"] == "val"]
        if not validation and all(row[f"Fold {fold}"] == "missing" for fold in range(5)):
            continue
        if len(validation) != 1:
            raise ValueError(f"Expected one validation fold for {row['ID']}")
        result[str(row["ID"])] = validation[0]
    return result


def patient_groups(tile_manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        str(patient): np.asarray(indices, dtype=np.int64)
        for patient, indices in tile_manifest.groupby("patient_id", sort=True).indices.items()
    }


def load_model(
    pathlupi_root: Path,
    checkpoint: Path,
    device: torch.device,
) -> torch.nn.Module:
    sys.path.insert(0, str(pathlupi_root))
    try:
        from inference import load_model as official_load_model
    finally:
        sys.path.pop(0)
    return official_load_model(checkpoint, True, device)


def predict_one(model: torch.nn.Module, bag: torch.Tensor) -> tuple[float, float]:
    with torch.inference_mode():
        outputs = model(x_path=bag)
    survival = outputs[1][0]
    return float(-survival.sum().item()), float(survival[-1].item())


def ordered_scoring_bag(
    *,
    source_index: int,
    context_indices: np.ndarray,
    variant_embedding: torch.Tensor,
    tile_embeddings: torch.Tensor,
) -> torch.Tensor:
    ordered = sorted(
        [(int(index), None) for index in context_indices]
        + [(int(source_index), variant_embedding)],
        key=lambda pair: pair[0],
    )
    rows = [
        tile_embeddings[index] if replacement is None else replacement
        for index, replacement in ordered
    ]
    return torch.stack(rows)


def evaluate_fold(
    *,
    fold: int,
    model: torch.nn.Module,
    tile_manifest: pd.DataFrame,
    clinical: pd.DataFrame,
    variant_manifest: pd.DataFrame,
    tile_embeddings: torch.Tensor,
    variant_embeddings: torch.Tensor,
    fold_map: dict[str, int],
    groups: dict[str, np.ndarray],
    tile_lookup: dict[str, int],
    bag_size: int,
    seed: int,
    ensemble_unassigned: bool,
    variant_risk_sum: np.ndarray,
    variant_probability_sum: np.ndarray,
    variant_prediction_count: np.ndarray,
    patient_rows: list[dict[str, Any]],
    max_sources: int | None,
    score_performance: bool = True,
) -> None:
    clinical_lookup = clinical.set_index("patient_id")
    if score_performance and not ensemble_unassigned:
        for patient, indices in groups.items():
            if fold_map.get(patient) != fold or len(indices) < bag_size:
                continue
            if patient not in clinical_lookup.index:
                continue
            record = clinical_lookup.loc[patient]
            if pd.isna(record.survival_time_days) or pd.isna(record.survival_event):
                continue
            selected = stable_rng(
                seed, "pathlupi-performance", patient, str(bag_size)
            ).choice(indices, size=bag_size, replace=False)
            bag = tile_embeddings[torch.as_tensor(np.sort(selected), device=tile_embeddings.device)]
            risk, probability = predict_one(model, bag)
            patient_rows.append(
                {
                    "patient_id": patient,
                    "official_fold": fold,
                    "bag_size": bag_size,
                    "risk_score": risk,
                    "last_bin_survival_probability": probability,
                    "survival_time_days": float(record.survival_time_days),
                    "survival_event": int(record.survival_event),
                }
            )

    source_groups = list(variant_manifest.groupby("source_tile_id", sort=True))
    if max_sources is not None:
        source_groups = source_groups[:max_sources]
    processed = 0
    for source_tile, rows in source_groups:
        patient = rows["patient_id_resolved"].iloc[0]
        if pd.isna(patient):
            continue
        patient = str(patient)
        assigned_fold = fold_map.get(patient)
        if ensemble_unassigned:
            if assigned_fold is not None:
                continue
        elif assigned_fold != fold:
            continue
        source_index = tile_lookup.get(str(source_tile))
        patient_indices = groups.get(patient)
        if source_index is None or patient_indices is None:
            continue
        candidates = patient_indices[patient_indices != source_index]
        if len(candidates) < bag_size - 1:
            continue
        context = stable_rng(
            seed, "pathlupi-score", patient, str(source_tile), str(bag_size)
        ).choice(candidates, size=bag_size - 1, replace=False)
        for row_index in rows.index.to_numpy(dtype=np.int64):
            bag = ordered_scoring_bag(
                source_index=source_index,
                context_indices=context,
                variant_embedding=variant_embeddings[row_index],
                tile_embeddings=tile_embeddings,
            )
            risk, probability = predict_one(model, bag)
            variant_risk_sum[row_index] += risk
            variant_probability_sum[row_index] += probability
            variant_prediction_count[row_index] += 1
        processed += 1
        if processed % 50 == 0:
            mode = "ensemble" if ensemble_unassigned else f"fold {fold}"
            print(f"[{mode}] {processed} source panels", flush=True)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    endpoint_root = args.endpoint_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    pathlupi_root = args.pathlupi_root.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    tile_manifest = pd.read_csv(endpoint_root / "tile_manifest.csv")
    variants = pd.read_csv(endpoint_root / "counterfactual_variant_manifest.csv").reset_index(drop=True)
    clinical = pd.read_csv(endpoint_root / "clinical_normalized.csv")
    source_patient = dict(
        zip(tile_manifest["tile_id"].astype(str), tile_manifest["patient_id"].astype(str))
    )
    variants["patient_id_resolved"] = variants["source_tile_id"].astype(str).map(source_patient)
    tile_values, variant_values = load_conch_cache(endpoint_root, tile_manifest, variants)
    tile_embeddings = torch.from_numpy(tile_values).to(device)
    variant_embeddings = torch.from_numpy(variant_values).to(device)
    folds = official_fold_map(pathlupi_root)
    groups = patient_groups(tile_manifest)
    tile_lookup = {
        tile_id: index
        for index, tile_id in enumerate(tile_manifest["tile_id"].astype(str))
    }

    variant_risk_sum = np.zeros(len(variants), dtype=np.float64)
    variant_probability_sum = np.zeros(len(variants), dtype=np.float64)
    variant_prediction_count = np.zeros(len(variants), dtype=np.int16)
    patient_rows: list[dict[str, Any]] = []
    for fold in range(5):
        print(f"\nLoading PathLUPI fold {fold} on {device}", flush=True)
        model = load_model(
            pathlupi_root,
            checkpoint_root / "survival" / "BRCA" / f"fold{fold}.pth.tar",
            device,
        )
        evaluate_fold(
            fold=fold,
            model=model,
            tile_manifest=tile_manifest,
            clinical=clinical,
            variant_manifest=variants,
            tile_embeddings=tile_embeddings,
            variant_embeddings=variant_embeddings,
            fold_map=folds,
            groups=groups,
            tile_lookup=tile_lookup,
            bag_size=args.bag_size,
            seed=args.seed,
            ensemble_unassigned=False,
            variant_risk_sum=variant_risk_sum,
            variant_probability_sum=variant_probability_sum,
            variant_prediction_count=variant_prediction_count,
            patient_rows=patient_rows,
            max_sources=args.max_sources,
        )
        evaluate_fold(
            fold=fold,
            model=model,
            tile_manifest=tile_manifest,
            clinical=clinical,
            variant_manifest=variants,
            tile_embeddings=tile_embeddings,
            variant_embeddings=variant_embeddings,
            fold_map=folds,
            groups=groups,
            tile_lookup=tile_lookup,
            bag_size=args.bag_size,
            seed=args.seed,
            ensemble_unassigned=True,
            variant_risk_sum=variant_risk_sum,
            variant_probability_sum=variant_probability_sum,
            variant_prediction_count=variant_prediction_count,
            patient_rows=patient_rows,
            max_sources=args.max_sources,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    patient_predictions = pd.DataFrame(patient_rows).sort_values("patient_id")
    performance = {
        "model_id": MODEL_ID,
        "model": DISPLAY_NAME,
        "endpoint": "Overall survival",
        "bag_size": args.bag_size,
        "patients": len(patient_predictions),
        "events": int(patient_predictions["survival_event"].sum()),
        "c_index": float(
            concordance_index(
                patient_predictions["survival_time_days"].to_numpy(float),
                patient_predictions["survival_event"].to_numpy(int),
                patient_predictions["risk_score"].to_numpy(float),
            )
        ),
    }
    patient_predictions.to_parquet(
        output_root / f"pathlupi_survival_fixedbag{args.bag_size}_predictions.parquet",
        index=False,
    )

    assigned = variants["patient_id_resolved"].map(folds).notna().to_numpy()
    expected_counts = np.where(assigned, 1, 5)
    valid = variant_prediction_count == expected_counts
    probabilities = np.full(len(variants), np.nan, dtype=np.float64)
    probabilities[valid] = (
        variant_probability_sum[valid] / variant_prediction_count[valid]
    )
    summary, details = summarize_experiments(
        variants,
        bernoulli_vectors(probabilities),
        valid,
        model=MODEL_ID,
        endpoint="Overall survival",
        bag_size=args.bag_size,
        max_panels=None,
    )
    summary.to_csv(output_root / "pathlupi_experiment_summary.csv", index=False)
    details.to_parquet(output_root / "pathlupi_pair_metrics.parquet", index=False)
    pd.DataFrame([performance]).to_csv(output_root / "pathlupi_performance.csv", index=False)

    row: dict[str, Any] = {
        "Task": "Overall survival",
        "Model": DISPLAY_NAME,
        "Performance": performance["c_index"],
    }
    for spec in EXPERIMENTS:
        match = (
            summary[summary["experiment"].eq(spec.experiment_id)]
            if "experiment" in summary.columns
            else pd.DataFrame()
        )
        row[spec.display_name] = (
            "N/A"
            if match.empty
            else f"{match.iloc[0].mean_tvd:.4f} / {match.iloc[0].flip_rate:.4f}"
        )
    row["BNR"] = bnr_for(summary) if "experiment" in summary.columns else math.nan

    base_table = pd.read_csv(args.base_results_root / "table4_revised.csv")
    merged_table = pd.concat([base_table, pd.DataFrame([row])], ignore_index=True)
    merged_table.to_csv(output_root / "table4_revised_with_pathlupi.csv", index=False)
    (output_root / "table4_revised_with_pathlupi.md").write_text(
        markdown_table(merged_table), encoding="utf-8"
    )
    audit = {
        "protocol": "released PathLUPI BRCA fold with one counterfactual in a deterministic fixed-size bag",
        "bag_size": args.bag_size,
        "device": str(device),
        "cache_baseline_shape": list(tile_values.shape),
        "cache_counterfactual_shape": list(variant_values.shape),
        "patients_performance": performance["patients"],
        "events_performance": performance["events"],
        "variants_scored": int(valid.sum()),
        "variants_unscored": int((~valid).sum()),
        "assigned_fold_variants": int(assigned.sum()),
        "ensemble_variants": int((~assigned & valid).sum()),
        "elapsed_minutes": (time.perf_counter() - started) / 60.0,
    }
    (output_root / "pathlupi_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print("\nPathLUPI fixed-bag row", flush=True)
    print(pd.DataFrame([row]).to_string(index=False), flush=True)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()


