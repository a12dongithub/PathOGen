#!/usr/bin/env python3
"""Evaluate released BRCA OTSurv folds on available original-UNI tile bags."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import hf_hub_download

from cpathogen.endpoints.clinical import normalize_patient_id
from cpathogen.endpoints.encoders import choose_device
from cpathogen.endpoints.jsonio import write_json, write_jsonl
from cpathogen.endpoints.pretrained import (
    baseline_manifest,
    checkpoint_state,
    load_or_extract_features,
    patient_bags,
    replacement_bag,
    survival_metrics_from_predictions,
)
from cpathogen.endpoints.variants import (
    discover_variant_manifests,
    normalize_variant_manifests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical-tsi", type=Path, required=True)
    parser.add_argument("--tile-manifest", type=Path, required=True)
    parser.add_argument("--otsurv-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    return parser.parse_args()


def _official_test_folds(repo: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for fold in range(5):
        path = (
            repo
            / "src"
            / "splits"
            / "survival"
            / f"TCGA_BRCA_overall_survival_k={fold}"
            / "test.csv"
        )
        frame = pd.read_csv(path)
        for patient in frame["case_id"]:
            normalized = normalize_patient_id(patient)
            if normalized in mapping:
                raise ValueError(
                    f"OTSurv patient {normalized} occurs in multiple test folds"
                )
            mapping[normalized] = fold
    return mapping


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    output = (
        args.output_root.expanduser().resolve() / "models" / "otsurv_uni_pretrained"
    )
    output.mkdir(parents=True, exist_ok=True)
    repo = args.otsurv_repo.expanduser().resolve()
    if not (repo / "src" / "mil_models" / "model_otsurv.py").is_file():
        raise FileNotFoundError(f"Not an OTSurv checkout: {repo}")
    sys.path.insert(0, str(repo / "src"))
    from mil_models.model_otsurv import OTSurv

    manifest = baseline_manifest(args.tile_manifest)
    embeddings = load_or_extract_features(
        "uni",
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
            "uni",
            variant_manifest,
            cache_dir=args.output_root / "embedding_cache",
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cache_tag="counterfactuals_pretrained",
        )
    fold_map = _official_test_folds(repo)
    rows = []
    variant_records = []
    for fold in range(5):
        checkpoint = hf_hub_download(
            repo_id="Y-Research-Group/OTSurv",
            filename=f"checkpoints/model_brca_fold{fold}.pth",
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = OTSurv(
            num_classes=1,
            patch_dim=1024,
            hidden_dim=256,
            num_prototypes=16,
            dropout_rate=0.25,
        )
        model.load_state_dict(checkpoint_state(payload), strict=True)
        model = model.to(device).eval()
        selected = sorted(patient for patient in bags if fold_map.get(patient) == fold)
        baseline_risk = {}
        with torch.inference_mode():
            for patient in selected:
                features = torch.from_numpy(bags[patient]).to(
                    device=device, dtype=torch.float32
                )
                logits = model.forward_no_loss([features, 10, 1_000])["logits"]
                value = float(logits.squeeze().cpu())
                baseline_risk[patient] = value
                rows.append(
                    {
                        "patient_id": patient,
                        "official_fold": fold,
                        "tile_count": len(features),
                        "log_risk": value,
                    }
                )
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
                                "model_id": "otsurv_uni_pretrained",
                                "endpoint": "overall_survival",
                                "scoring_fold": fold,
                                "status": "bag_replacement_unavailable",
                                "prediction": None,
                                "baseline_prediction": None,
                            }
                        )
                        continue
                    features = torch.from_numpy(replaced).to(
                        device=device, dtype=torch.float32
                    )
                    log_risk = float(
                        model.forward_no_loss([features, 10, 1_000])["logits"]
                        .squeeze()
                        .cpu()
                    )
                    variant_records.append(
                        {
                            **variant.to_dict(),
                            "schema_version": 1,
                            "model_id": "otsurv_uni_pretrained",
                            "endpoint": "overall_survival",
                            "scoring_fold": fold,
                            "status": "ok",
                            "prediction": {"log_risk": log_risk},
                            "baseline_prediction": {
                                "log_risk": baseline_risk[variant["patient_id"]]
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
        risk_column="log_risk",
        model="OTSurv + original UNI (released BRCA folds)",
        endpoint_alignment=(
            "Checkpoint evaluation config names dss_survival_days, while this run evaluates "
            "Clinical.tsi overall survival; interpret as transfer, not exact endpoint reproduction."
        ),
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
                    "model_id": "otsurv_uni_pretrained",
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
        f"OTSurv: {len(predictions)} official test-fold patient predictions", flush=True
    )


if __name__ == "__main__":
    main()
