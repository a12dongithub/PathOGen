#!/usr/bin/env python3
"""Score 0/90/180/270-degree rotations with corrected endpoint protocols."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from run_local_xai_rerun import (
    DISPLAY_MODELS,
    CoxProbe,
    bernoulli_vectors,
    load_fold_map,
    predict_fixed_bag_variants,
    predict_tile_variants,
    summarize_experiments,
)

import __main__

# The original heads were serialized while run_local_xai_rerun.py executed as
# __main__, so expose the class under that name for backwards-compatible load.
__main__.CoxProbe = CoxProbe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-root", type=Path, required=True)
    parser.add_argument("--base-results-root", type=Path, required=True)
    parser.add_argument("--virchow-results-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=("resnet50", "ctranspath", "uni2h", "virchow2"),
    )
    parser.add_argument("--bag-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_embeddings(
    endpoint_root: Path,
    model: str,
    tile_manifest: pd.DataFrame,
    rotations: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    cache_root = endpoint_root / "embedding_cache"
    tile_cache = np.load(cache_root / f"{model}_tiles.npz", allow_pickle=False)
    rotation_cache = np.load(
        cache_root / f"{model}_rotation.npz", allow_pickle=False
    )
    if not np.array_equal(
        tile_cache["tile_ids"].astype(str),
        tile_manifest["tile_id"].astype(str).to_numpy(),
    ):
        raise ValueError(f"{model}: real cache does not match tile manifest")
    if not np.array_equal(
        rotation_cache["variant_ids"].astype(str),
        rotations["variant_id"].astype(str).to_numpy(),
    ):
        raise ValueError(f"{model}: rotation cache does not match rotation manifest")
    tiles = tile_cache["embeddings"].astype(np.float32, copy=False)
    rotated = rotation_cache["embeddings"].astype(np.float32, copy=False)
    if not np.isfinite(tiles).all() or not np.isfinite(rotated).all():
        raise ValueError(f"{model}: non-finite embeddings")
    return tiles, rotated


def main() -> None:
    args = parse_args()
    endpoint_root = args.endpoint_root.resolve()
    base_results = args.base_results_root.resolve()
    virchow_results = (
        args.virchow_results_root.resolve()
        if args.virchow_results_root
        else base_results
    )
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
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
    pam50_folds = load_fold_map(
        endpoint_root / "models" / "resnet50" / "pam50_patient_oof_predictions.csv"
    )
    survival_folds = load_fold_map(
        endpoint_root
        / "models"
        / "resnet50"
        / "survival_patient_oof_predictions.csv"
    )

    summaries = []
    details = []
    for model in args.models:
        if model not in DISPLAY_MODELS:
            raise ValueError(f"Unknown model {model}")
        head_root = virchow_results if model == "virchow2" else base_results
        tiles, rotated = load_embeddings(endpoint_root, model, tile_manifest, rotations)
        pam_heads = joblib.load(head_root / f"{model}_pam50_tile_heads.joblib")
        pam_probabilities, pam_valid = predict_tile_variants(
            rotations, rotated, pam_heads, pam50_folds
        )
        pam_summary, pam_detail = summarize_experiments(
            rotations,
            pam_probabilities,
            pam_valid,
            model=model,
            endpoint="PAM50",
            bag_size=None,
            max_panels=None,
        )
        summaries.append(pam_summary)
        details.append(pam_detail)

        survival_heads = joblib.load(
            head_root / f"{model}_survival_fixedbag{args.bag_size}_heads.joblib"
        )
        survival_probability, survival_valid = predict_fixed_bag_variants(
            rotations,
            rotated,
            tile_manifest,
            tiles,
            survival_heads,
            survival_folds,
            bag_size=args.bag_size,
            seed=args.seed,
        )
        survival_summary, survival_detail = summarize_experiments(
            rotations,
            bernoulli_vectors(survival_probability),
            survival_valid,
            model=model,
            endpoint="Overall survival",
            bag_size=args.bag_size,
            max_panels=None,
        )
        summaries.append(survival_summary)
        details.append(survival_detail)
        print(
            f"[{model}] rotation panels: PAM50={pam_summary.iloc[0].tiles}, "
            f"survival={survival_summary.iloc[0].tiles}",
            flush=True,
        )

    summary = pd.concat(summaries, ignore_index=True)
    detail = pd.concat(details, ignore_index=True)
    summary.to_csv(output / "rotation_experiment_summary.csv", index=False)
    detail.to_parquet(output / "rotation_pair_metrics.parquet", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()


