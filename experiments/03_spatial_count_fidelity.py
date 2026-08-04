#!/usr/bin/env python
"""Correlate input source-cell counts with CellViT++ counts in generated H&E."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.fidelity.constants import CELL_NAMES_WITH_TOTAL
from experiments.fidelity.data import load_cells
from experiments.fidelity.spatial import cell_counts
from experiments.fidelity.spatial_workflow import ensure_spatial_case
from experiments.fidelity.statistics import benjamini_hochberg, spearman_with_bootstrap
from experiments.fidelity.workflow import (
    ExperimentRuntime,
    add_common_arguments,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


def analyze(frame: pd.DataFrame, bootstrap: int, seed: int) -> pd.DataFrame:
    summaries = []
    for index, cell_type in enumerate(CELL_NAMES_WITH_TOTAL):
        subset = frame[frame["cell_type"] == cell_type]
        result = spearman_with_bootstrap(
            subset["input_count"].to_numpy(float),
            subset["generated_count"].to_numpy(float),
            bootstrap=bootstrap,
            seed=seed + index,
        )
        difference = subset["generated_count"].to_numpy(float) - subset[
            "input_count"
        ].to_numpy(float)
        summaries.append(
            {
                "cell_type": cell_type,
                "n": result.n,
                "spearman_rho": result.rho,
                "p_value": result.p_value,
                "ci_low": result.ci_low,
                "ci_high": result.ci_high,
                "mean_input_count": float(subset["input_count"].mean()),
                "mean_generated_count": float(subset["generated_count"].mean()),
                "median_count_error": float(np.median(difference)),
                "median_absolute_error": float(np.median(np.abs(difference))),
            }
        )
    summary = pd.DataFrame(summaries)
    summary["p_fdr_bh"] = benjamini_hochberg(summary["p_value"].tolist())
    return summary


def main() -> None:
    args = parse_args()
    runtime = ExperimentRuntime(args)
    stems = runtime.selected_stems()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "spatial_case_plan.json",
        {"stems": stems, "seed": args.seed, "same_inputs_as_generation": True},
    )
    if args.dry_run:
        for stem in stems:
            source_count = len(load_cells(runtime.catalog.sample(stem).geojson_path))
            if source_count < 1:
                raise RuntimeError(f"Source GeoJSON has no recognized cells: {stem}")
        print(f"Dry run passed for {len(stems)} aligned spatial cases")
        return

    rows = []
    try:
        for case_index, stem in enumerate(stems, start=1):
            sample = runtime.catalog.sample(stem)
            image_path, geojson_path, generation_metadata = ensure_spatial_case(
                runtime, stem, args.seed
            )
            input_counts = cell_counts(load_cells(sample.geojson_path))
            generated_counts = cell_counts(load_cells(geojson_path))
            for cell_type in CELL_NAMES_WITH_TOTAL:
                rows.append(
                    {
                        "stem": stem,
                        "cell_type": cell_type,
                        "input_count": input_counts[cell_type],
                        "generated_count": generated_counts[cell_type],
                        "generated_image": str(image_path),
                        "cellvit_geojson": str(geojson_path),
                        "generation_seconds": generation_metadata.get("seconds", np.nan),
                        "accepted": generation_metadata.get("accepted", True),
                    }
                )
            print(f"[{case_index}/{len(stems)}] completed {stem}", flush=True)
    finally:
        runtime.close()
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "spatial_count_pairs.csv", index=False)
    summary = analyze(frame, args.bootstrap, args.seed)
    summary.to_csv(args.output_dir / "spatial_count_spearman.csv", index=False)
    write_json(
        args.output_dir / "spatial_count_manifest.json",
        {
            "experiment": "spatial_count_fidelity",
            "num_images": len(stems),
            "input_counts": "exact centroids from source GeoJSON used to make conditioning maps",
            "generated_counts": "CellViT++ detections on generated H&E",
            "cell_types": CELL_NAMES_WITH_TOTAL,
            "steps": args.steps,
            "guidance_hook": args.guidance_hook,
        },
    )
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
