#!/usr/bin/env python3
"""Run the released CPMP MammaPrint-risk checkpoint on available original-UNI bags."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from cpathogen.endpoints.clinical import load_clinical_matrix
from cpathogen.endpoints.encoders import choose_device
from cpathogen.endpoints.jsonio import write_json, write_jsonl
from cpathogen.endpoints.pretrained import (
    baseline_manifest,
    checkpoint_state,
    load_or_extract_features,
    patient_bags,
    patient_coordinates,
    replacement_bag,
)
from cpathogen.endpoints.variants import (
    discover_variant_manifests,
    normalize_variant_manifests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical-tsi", type=Path, required=True)
    parser.add_argument("--tile-manifest", type=Path, required=True)
    parser.add_argument("--cpmp-repo", type=Path, required=True)
    parser.add_argument("--cpmp-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    output = args.output_root.expanduser().resolve() / "models" / "cpmp_uni_pretrained"
    output.mkdir(parents=True, exist_ok=True)
    repo = args.cpmp_repo.expanduser().resolve()
    if not (repo / "models" / "transformer.py").is_file():
        raise FileNotFoundError(f"Not a CPMP checkout: {repo}")
    if not args.cpmp_checkpoint.is_file():
        raise FileNotFoundError(args.cpmp_checkpoint)
    sys.path.insert(0, str(repo))
    from models.transformer import Transformer

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
    coordinates = patient_coordinates(manifest)
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
    model = Transformer(
        num_classes=1,
        input_dim=1024,
        depth=1,
        heads=4,
        dim_head=64,
        hidden_dim=512,
        pool="cls",
        dropout=0.5,
        emb_dropout=0.0,
        pos_enc=None,
        agent_n=1,
    )
    payload = torch.load(args.cpmp_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint_state(payload), strict=True)
    model = model.to(device).eval()
    rows = []
    baseline_low: dict[str, float] = {}
    with torch.inference_mode():
        for patient, bag in sorted(bags.items()):
            features = torch.from_numpy(bag).to(device=device, dtype=torch.float32)
            coords = torch.from_numpy(coordinates[patient]).to(
                device=device, dtype=torch.float32
            )
            probability_low, _, _ = model(features, coords=coords, register_hook=False)
            low = float(probability_low.squeeze().cpu())
            baseline_low[patient] = low
            rows.append(
                {
                    "patient_id": patient,
                    "tile_count": len(features),
                    "cpmp_probability_low_mammaprint_risk": low,
                    "cpmp_probability_high_mammaprint_risk": 1.0 - low,
                    "cpmp_repository_threshold_label": "Low" if low > 0.571 else "High",
                }
            )
        variant_records = []
        if variants is not None and variant_embeddings is not None:
            for index, variant in variants.iterrows():
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
                            "model_id": "cpmp_uni_pretrained",
                            "endpoint": "MammaPrint_recurrence_risk",
                            "scoring_fold": None,
                            "status": "bag_replacement_unavailable",
                            "prediction": None,
                            "baseline_prediction": None,
                        }
                    )
                    continue
                features = torch.from_numpy(replaced).to(
                    device=device, dtype=torch.float32
                )
                coords = torch.from_numpy(coordinates[variant["patient_id"]]).to(
                    device=device, dtype=torch.float32
                )
                low = float(
                    model(features, coords=coords, register_hook=False)[0]
                    .squeeze()
                    .cpu()
                )
                base = baseline_low[variant["patient_id"]]
                variant_records.append(
                    {
                        **variant.to_dict(),
                        "schema_version": 1,
                        "model_id": "cpmp_uni_pretrained",
                        "endpoint": "MammaPrint_recurrence_risk",
                        "scoring_fold": None,
                        "status": "ok",
                        "prediction": {
                            "probability_low": low,
                            "probability_high": 1.0 - low,
                        },
                        "baseline_prediction": {
                            "probability_low": base,
                            "probability_high": 1.0 - base,
                        },
                    }
                )
    predictions = pd.DataFrame(rows).merge(
        load_clinical_matrix(args.clinical_tsi),
        on="patient_id",
        how="left",
        validate="one_to_one",
    )
    predictions.to_csv(output / "recurrence_risk_patient_predictions.csv", index=False)
    predictions.to_parquet(
        output / "recurrence_risk_patient_predictions.parquet", index=False
    )
    metrics = {
        "schema_version": 1,
        "model": "CPMP + original UNI (released t0f0 checkpoint)",
        "endpoint": "MammaPrint recurrence-risk class",
        "patients_predicted": len(predictions),
        "performance_evaluable": False,
        "auc": None,
        "f1": None,
        "accuracy": None,
        "reason": (
            "Clinical.tsi has no patient-level MammaPrint Low/High label and no recurrence "
            "time/event. Overall survival must not be substituted for recurrence ground truth."
        ),
        "score_semantics": (
            "The repository maps Low to 1 and High to 0; therefore the sigmoid output is "
            "stored as probability_low and 1-output as probability_high."
        ),
        "threshold": 0.571,
        "limitation": (
            "The 512 dataset is an available tile subset rather than CPMP's original full-WSI "
            "tiling and tissue-selection pipeline."
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "recurrence_risk_metrics.json", metrics)
    if variant_records:
        write_jsonl(output / "counterfactual_predictions.jsonl", variant_records)
    print(
        f"CPMP: wrote {len(predictions)} scores; performance not evaluable", flush=True
    )


if __name__ == "__main__":
    main()
