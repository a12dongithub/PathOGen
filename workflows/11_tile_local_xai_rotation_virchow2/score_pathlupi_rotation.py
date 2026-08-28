#!/usr/bin/env python3
"""Score rotation nuisance with official PathLUPI BRCA fixed-size bags."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_local_xai_rerun import bernoulli_vectors, summarize_experiments
from run_pathlupi_fixedbag import (
    MODEL_ID,
    SURVIVAL_BIN_INDEX,
    SURVIVAL_INTERVAL_MONTHS,
    evaluate_fold,
    load_model,
    official_fold_map,
    patient_groups,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-root", type=Path, required=True)
    parser.add_argument("--pathlupi-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bag-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    endpoint_root = args.endpoint_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    tile_manifest = pd.read_csv(endpoint_root / "tile_manifest.csv")
    rotations = pd.read_csv(endpoint_root / "rotation_variant_manifest.csv").reset_index(
        drop=True
    )
    source_patient = dict(
        zip(tile_manifest["tile_id"].astype(str), tile_manifest["patient_id"].astype(str))
    )
    rotations["patient_id_resolved"] = rotations["source_tile_id"].astype(str).map(
        source_patient
    )
    cache_root = endpoint_root / "embedding_cache"
    with np.load(cache_root / "conch_pretrained_bags.npz", allow_pickle=False) as cache:
        if not np.array_equal(
            cache["tile_ids"].astype(str), tile_manifest["tile_id"].astype(str)
        ):
            raise ValueError("CONCH real cache does not match tile manifest")
        tile_values = cache["embeddings"].astype(np.float32)
    with np.load(cache_root / "conch_rotation.npz", allow_pickle=False) as cache:
        if not np.array_equal(
            cache["variant_ids"].astype(str), rotations["variant_id"].astype(str)
        ):
            raise ValueError("CONCH rotation cache does not match rotation manifest")
        rotation_values = cache["embeddings"].astype(np.float32)
    if not np.isfinite(tile_values).all() or not np.isfinite(rotation_values).all():
        raise ValueError("CONCH embeddings contain non-finite values")

    tiles = torch.from_numpy(tile_values).to(device)
    rotated = torch.from_numpy(rotation_values).to(device)
    folds = official_fold_map(args.pathlupi_root.resolve())
    groups = patient_groups(tile_manifest)
    tile_lookup = {
        tile_id: index
        for index, tile_id in enumerate(tile_manifest["tile_id"].astype(str))
    }
    probability_sum = np.zeros(len(rotations), dtype=np.float64)
    risk_sum = np.zeros(len(rotations), dtype=np.float64)
    prediction_count = np.zeros(len(rotations), dtype=np.int16)
    for fold in range(5):
        print(f"Loading PathLUPI fold {fold} on {device}", flush=True)
        model = load_model(
            args.pathlupi_root.resolve(),
            args.checkpoint_root.resolve()
            / "survival"
            / "BRCA"
            / f"fold{fold}.pth.tar",
            device,
        )
        for ensemble in (False, True):
            evaluate_fold(
                fold=fold,
                model=model,
                tile_manifest=tile_manifest,
                clinical=pd.DataFrame(columns=["patient_id"]),
                variant_manifest=rotations,
                tile_embeddings=tiles,
                variant_embeddings=rotated,
                fold_map=folds,
                groups=groups,
                tile_lookup=tile_lookup,
                bag_size=args.bag_size,
                seed=args.seed,
                ensemble_unassigned=ensemble,
                variant_risk_sum=risk_sum,
                variant_probability_sum=probability_sum,
                variant_prediction_count=prediction_count,
                patient_rows=[],
                max_sources=None,
                score_performance=False,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assigned = rotations["patient_id_resolved"].map(folds).notna().to_numpy()
    expected = np.where(assigned, 1, 5)
    valid = prediction_count == expected
    probabilities = np.full(len(rotations), np.nan, dtype=np.float64)
    probabilities[valid] = probability_sum[valid] / prediction_count[valid]
    summary, details = summarize_experiments(
        rotations,
        bernoulli_vectors(probabilities),
        valid,
        model=MODEL_ID,
        endpoint="Overall survival",
        bag_size=args.bag_size,
        max_panels=None,
    )
    summary.to_csv(output / "pathlupi_rotation_summary.csv", index=False)
    details.to_parquet(output / "pathlupi_rotation_pair_metrics.parquet", index=False)
    audit = {
        "variants": len(rotations),
        "variants_scored": int(valid.sum()),
        "complete_panels": int(summary.iloc[0].tiles),
        "patients": int(summary.iloc[0].patients),
        "bag_size": args.bag_size,
        "survival_probability_bin_index": SURVIVAL_BIN_INDEX,
        "survival_probability_interval_months": list(SURVIVAL_INTERVAL_MONTHS),
    }
    (output / "pathlupi_rotation_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

