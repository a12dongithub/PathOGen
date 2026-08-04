#!/usr/bin/env python
"""Paired PathOGen morphology interventions followed by CellViT++ and Spearman tests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.fidelity.constants import MORPH_FEATURES, MORPH_MEAN_FEATURES
from experiments.fidelity.data import load_cells
from experiments.fidelity.guidance import GenerationContext
from experiments.fidelity.measurements import morphology_measurements
from experiments.fidelity.statistics import benjamini_hochberg, spearman_with_bootstrap
from experiments.fidelity.workflow import (
    ExperimentRuntime,
    add_common_arguments,
    deterministic_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument(
        "--features",
        nargs="+",
        default=MORPH_MEAN_FEATURES,
        help="Morphology coordinates to increase independently (default: all eight means)",
    )
    parser.add_argument("--quantile-shift", type=float, default=0.20)
    parser.add_argument("--range-lower-quantile", type=float, default=0.01)
    parser.add_argument("--range-upper-quantile", type=float, default=0.99)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def build_plan(runtime: ExperimentRuntime, args: argparse.Namespace) -> list[dict]:
    unknown = sorted(set(args.features) - set(MORPH_FEATURES))
    if unknown:
        raise ValueError(f"Unknown morphology features: {unknown}")
    if args.stems:
        candidates = runtime.catalog.select(len(args.stems), args.seed, args.stems)
    else:
        candidates = runtime.catalog.select(len(runtime.catalog.stems), args.seed)

    plan: list[dict] = []
    selected = 0
    failures: dict[str, int] = {feature: 0 for feature in args.features}
    for stem in candidates:
        sample = runtime.catalog.sample(stem)
        interventions = []
        eligible = True
        for feature in args.features:
            try:
                changed, details = runtime.catalog.increase_feature(
                    sample.morphology,
                    feature,
                    args.quantile_shift,
                    args.range_lower_quantile,
                    args.range_upper_quantile,
                )
            except ValueError:
                failures[feature] += 1
                eligible = False
                break
            interventions.append(
                {
                    "feature": feature,
                    "morphology": changed.astype(float).tolist(),
                    **details,
                }
            )
        if not eligible:
            if args.stems:
                raise ValueError(
                    f"Explicitly requested stem {stem} cannot be increased for every feature "
                    "within the configured empirical range"
                )
            continue
        paired_seed = deterministic_seed(args.seed, stem, "paired_morphology")
        plan.append(
            {
                "stem": stem,
                "seed": paired_seed,
                "baseline_morphology": sample.morphology.astype(float).tolist(),
                "interventions": interventions,
            }
        )
        selected += 1
        if selected == args.num_images:
            break
    if len(plan) != args.num_images:
        raise RuntimeError(
            f"Only {len(plan)} cases were eligible for all interventions; requested "
            f"{args.num_images}. Failures={failures}"
        )
    return plan


def measure_artifact(
    runtime: ExperimentRuntime,
    stem: str,
    condition_id: str,
    morphology: np.ndarray,
    seed: int,
) -> dict:
    sample = runtime.catalog.sample(stem)
    artifact_name = f"{stem}__{condition_id}"
    context = GenerationContext(
        stem=stem,
        condition_id=condition_id,
        spatial_map=runtime.catalog.load_spatial(sample.spatial_path),
        morphology=morphology,
        seed=seed,
        metadata={"experiment": "morphology_fidelity"},
    )
    image_path, generation_metadata = runtime.ensure_generated(context, artifact_name)
    geojson_path = runtime.ensure_cellvit(image_path, artifact_name)
    measurements = morphology_measurements(
        Image.open(image_path).convert("RGB"), load_cells(geojson_path)
    )
    row = {
        "stem": stem,
        "condition_id": condition_id,
        "seed": seed,
        "image_path": str(image_path),
        "cellvit_geojson": str(geojson_path),
        "generation_seconds": generation_metadata.get("seconds", float("nan")),
        "accepted": generation_metadata.get("accepted", True),
        "guidance_score": generation_metadata.get("guidance_score"),
    }
    row.update({f"input_{name}": float(value) for name, value in zip(MORPH_FEATURES, morphology)})
    row.update({f"measured_{name}": value for name, value in measurements.items()})
    return row


def analyze(frame: pd.DataFrame, features: list[str], bootstrap: int, seed: int) -> pd.DataFrame:
    baseline = frame[frame["condition_id"] == "baseline"].set_index("stem")
    summaries = []
    for feature_index, feature in enumerate(features):
        changed = frame[frame["condition_id"] == f"increase__{feature}"].set_index("stem")
        stems = sorted(set(baseline.index) & set(changed.index))
        base = baseline.loc[stems]
        target = changed.loc[stems]
        base_input = base[f"input_{feature}"].to_numpy(float)
        target_input = target[f"input_{feature}"].to_numpy(float)
        base_output = base[f"measured_{feature}"].to_numpy(float)
        target_output = target[f"measured_{feature}"].to_numpy(float)
        pooled = spearman_with_bootstrap(
            np.concatenate([base_input, target_input]),
            np.concatenate([base_output, target_output]),
            bootstrap=bootstrap,
            seed=seed + feature_index * 10,
            groups=np.asarray(stems + stems),
        )
        target_only = spearman_with_bootstrap(
            target_input,
            target_output,
            bootstrap=bootstrap,
            seed=seed + feature_index * 10 + 1,
        )
        input_delta = target_input - base_input
        output_delta = target_output - base_output
        delta = spearman_with_bootstrap(
            input_delta,
            output_delta,
            bootstrap=bootstrap,
            seed=seed + feature_index * 10 + 2,
        )
        finite_delta = output_delta[np.isfinite(output_delta)]
        summaries.append(
            {
                "feature": feature,
                "pairs": len(stems),
                "valid_output_pairs": int(np.isfinite(base_output * target_output).sum()),
                "pooled_rho": pooled.rho,
                "pooled_p_value": pooled.p_value,
                "pooled_ci_low": pooled.ci_low,
                "pooled_ci_high": pooled.ci_high,
                "perturbed_only_rho": target_only.rho,
                "perturbed_only_p_value": target_only.p_value,
                "delta_rho": delta.rho,
                "delta_p_value": delta.p_value,
                "median_input_delta": float(np.nanmedian(input_delta)),
                "median_output_delta": (
                    float(np.median(finite_delta)) if len(finite_delta) else float("nan")
                ),
                "direction_accuracy": (
                    float(np.mean(finite_delta > 0)) if len(finite_delta) else float("nan")
                ),
            }
        )
    result = pd.DataFrame(summaries)
    result["pooled_p_fdr_bh"] = benjamini_hochberg(result["pooled_p_value"].tolist())
    return result


def main() -> None:
    args = parse_args()
    runtime = ExperimentRuntime(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranges = runtime.catalog.feature_ranges(
        args.range_lower_quantile, args.range_upper_quantile
    )
    ranges.to_csv(args.output_dir / "feature_ranges.csv", index=False)
    plan = build_plan(runtime, args)
    write_json(args.output_dir / "condition_plan.json", plan)
    if args.dry_run:
        print(f"Dry run passed: {len(plan)} cases x {len(args.features)} interventions")
        return

    rows = []
    try:
        for case_index, item in enumerate(plan, start=1):
            stem = item["stem"]
            seed = int(item["seed"])
            baseline = np.asarray(item["baseline_morphology"], dtype=np.float32)
            rows.append(measure_artifact(runtime, stem, "baseline", baseline, seed))
            for intervention in item["interventions"]:
                feature = intervention["feature"]
                morphology = np.asarray(intervention["morphology"], dtype=np.float32)
                rows.append(
                    measure_artifact(
                        runtime, stem, f"increase__{feature}", morphology, seed
                    )
                )
            print(f"[{case_index}/{len(plan)}] completed {stem}", flush=True)
    finally:
        runtime.close()

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "measurements.csv", index=False)
    summary = analyze(frame, args.features, args.bootstrap, args.seed)
    summary.to_csv(args.output_dir / "spearman_summary.csv", index=False)
    write_json(
        args.output_dir / "manifest.json",
        {
            "experiment": "morphology_fidelity",
            "num_images": len(plan),
            "features": args.features,
            "same_seed_within_pair": True,
            "quantile_shift": args.quantile_shift,
            "allowed_quantile_range": [
                args.range_lower_quantile,
                args.range_upper_quantile,
            ],
            "steps": args.steps,
            "spatial_strength": args.spatial_strength,
            "guidance_hook": args.guidance_hook,
        },
    )
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
