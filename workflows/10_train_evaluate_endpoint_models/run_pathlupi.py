#!/usr/bin/env python3
"""Evaluate released BRCA PathLUPI survival folds on available CONCH tile bags."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

from cpathogen.endpoints.encoders import choose_device
from cpathogen.endpoints.jsonio import write_json, write_jsonl
from cpathogen.endpoints.pretrained import (
    baseline_manifest,
    checkpoint_state,
    load_or_extract_features,
    patient_bags,
    replacement_bag,
    survival_metrics_from_predictions,
    wide_fold_mapping,
)
from cpathogen.endpoints.variants import (
    discover_variant_manifests,
    normalize_variant_manifests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical-tsi", type=Path, required=True)
    parser.add_argument("--tile-manifest", type=Path, required=True)
    parser.add_argument("--pathlupi-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    return parser.parse_args()


def _build_model(network_class, state: dict[str, torch.Tensor], device: torch.device):
    pathway_sizes = [
        value.shape[0]
        for key, value in state.items()
        if key.startswith("recon_net.sig_reconstruct_net.")
        and key.endswith(".4.weight")
    ]
    path_size = state["wsi_net.0.weight"].shape[1]
    projected_size = state["wsi_net.0.weight"].shape[0]
    classes = state["classifier.weight"].shape[0]
    tokens = state["recon_net.token_generator.2.weight"].shape[0] // projected_size
    model = network_class(
        omic_sizes=pathway_sizes,
        n_classes=classes,
        path_size=path_size,
        path_proj_size=projected_size,
        num_tokens=tokens,
        region_num=tokens,
        survival=True,
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    output = (
        args.output_root.expanduser().resolve() / "models" / "pathlupi_conch_pretrained"
    )
    output.mkdir(parents=True, exist_ok=True)
    repo = args.pathlupi_repo.expanduser().resolve()
    if not (repo / "models" / "PathLUPI" / "network.py").is_file():
        raise FileNotFoundError(f"Not a PathLUPI checkout: {repo}")
    sys.path.insert(0, str(repo))
    from models.PathLUPI.network import PathLUPI

    manifest = baseline_manifest(args.tile_manifest)
    embeddings = load_or_extract_features(
        "conch",
        manifest,
        cache_dir=args.output_root / "embedding_cache",
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    bags = patient_bags(manifest, embeddings)
    variants = None
    variant_embeddings = None
    if args.variant_root:
        variants = normalize_variant_manifests(
            discover_variant_manifests(args.variant_root)
        )
        variant_manifest = variants.rename(columns={"variant_id": "tile_id"})
        variant_embeddings = load_or_extract_features(
            "conch",
            variant_manifest,
            cache_dir=args.output_root / "embedding_cache",
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cache_tag="counterfactuals_pretrained",
        )
    fold_map = wide_fold_mapping(
        repo / "splits" / "survival" / "BRCA_Splits.csv",
        patient_column="ID",
        held_out_values=("val", "test"),
    )
    rows = []
    variant_records = []
    for fold in range(5):
        checkpoint = hf_hub_download(
            repo_id="peterjin0703/PathLUPI",
            filename=f"survival/BRCA/fold{fold}.pth.tar",
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = _build_model(PathLUPI, checkpoint_state(payload), device)
        selected = sorted(patient for patient in bags if fold_map.get(patient) == fold)
        baseline_survival: dict[str, np.ndarray] = {}
        with torch.inference_mode():
            for patient in selected:
                bag = bags[patient]
                if len(bag) == 1:
                    bag = np.repeat(bag, 2, axis=0)
                features = torch.from_numpy(bag).to(device=device, dtype=torch.float32)
                _hazards, survival, *_ = model(x_path=features)
                survival_values = survival[0].float().cpu().numpy()
                baseline_survival[patient] = survival_values
                row = {
                    "patient_id": patient,
                    "official_fold": fold,
                    "tile_count": len(features),
                    "risk_score": float(-survival_values.sum()),
                }
                for index, value in enumerate(survival_values):
                    row[f"discrete_survival_bin_{index}"] = float(value)
                rows.append(row)
            if variants is not None and variant_embeddings is not None:
                fold_variants = variants[variants["patient_id"].map(fold_map) == fold]
                for index, variant in fold_variants.iterrows():
                    replaced = replacement_bag(
                        manifest,
                        bags,
                        patient_id=variant["patient_id"],
                        source_tile_id=variant["source_tile_id"],
                        replacement=variant_embeddings[index],
                    )
                    if replaced is None:
                        variant_records.append(
                            {
                                **variant.to_dict(),
                                "schema_version": 1,
                                "model_id": "pathlupi_conch_pretrained",
                                "endpoint": "overall_survival",
                                "scoring_fold": fold,
                                "status": "bag_replacement_unavailable",
                                "prediction": None,
                                "baseline_prediction": None,
                            }
                        )
                        continue
                    if len(replaced) == 1:
                        replaced = np.repeat(replaced, 2, axis=0)
                    replacement_tensor = torch.from_numpy(replaced).to(
                        device=device, dtype=torch.float32
                    )
                    replacement_survival = (
                        model(x_path=replacement_tensor)[1][0].float().cpu().numpy()
                    )
                    original_survival = baseline_survival[variant["patient_id"]]
                    variant_records.append(
                        {
                            **variant.to_dict(),
                            "schema_version": 1,
                            "model_id": "pathlupi_conch_pretrained",
                            "endpoint": "overall_survival",
                            "scoring_fold": fold,
                            "status": "ok",
                            "prediction": {
                                "risk_score": float(-replacement_survival.sum()),
                                "discrete_survival": replacement_survival.tolist(),
                            },
                            "baseline_prediction": {
                                "risk_score": float(-original_survival.sum()),
                                "discrete_survival": original_survival.tolist(),
                            },
                        }
                    )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    predictions = pd.DataFrame(rows).sort_values("patient_id", kind="stable")
    predictions, metrics = survival_metrics_from_predictions(
        predictions,
        args.clinical_tsi,
        risk_column="risk_score",
        model="PathLUPI + CONCH (released BRCA survival folds)",
        endpoint_alignment="Released BRCA survival endpoint evaluated against Clinical.tsi overall survival.",
    )
    predictions.to_csv(output / "survival_patient_predictions.csv", index=False)
    predictions.to_parquet(output / "survival_patient_predictions.parquet", index=False)
    write_json(output / "survival_metrics.json", metrics)
    if variants is not None:
        missing_fold = variants["patient_id"].map(fold_map).isna()
        for _, variant in variants[missing_fold].iterrows():
            variant_records.append(
                {
                    **variant.to_dict(),
                    "schema_version": 1,
                    "model_id": "pathlupi_conch_pretrained",
                    "endpoint": "overall_survival",
                    "scoring_fold": None,
                    "status": "patient_not_in_official_folds",
                    "prediction": None,
                    "baseline_prediction": None,
                }
            )
    if variant_records:
        write_jsonl(output / "counterfactual_predictions.jsonl", variant_records)
    print(
        f"PathLUPI: {len(predictions)} official held-out patient predictions",
        flush=True,
    )


if __name__ == "__main__":
    main()
