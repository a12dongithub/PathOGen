#!/usr/bin/env python
"""Measure class-specific spatial-coordinate fidelity of generated H&E images."""

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
from experiments.fidelity.spatial import coordinate_metrics
from experiments.fidelity.spatial_workflow import ensure_spatial_case
from experiments.fidelity.statistics import benjamini_hochberg, spearman_with_bootstrap
from experiments.fidelity.workflow import (
    ExperimentRuntime,
    add_common_arguments,
    write_json,
)


def finite_median(values: pd.Series) -> float:
    finite = values[np.isfinite(values.to_numpy(dtype=float))]
    return float(finite.median()) if len(finite) else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--max-match-distance", type=float, default=32.0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser.parse_args()


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
        if args.grid_size < 2 or args.max_match_distance <= 0:
            raise ValueError("grid-size must be >=2 and max-match-distance must be positive")
        for stem in stems:
            sample = runtime.catalog.sample(stem)
            if not load_cells(sample.geojson_path):
                raise RuntimeError(f"Source GeoJSON has no recognized cells: {stem}")
            runtime.catalog.load_spatial(sample.spatial_path)
        print(f"Dry run passed for {len(stems)} coordinate-fidelity cases")
        return

    per_image_rows = []
    match_rows = []
    grids: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {
        cell_type: [] for cell_type in CELL_NAMES_WITH_TOTAL
    }
    try:
        for case_index, stem in enumerate(stems, start=1):
            sample = runtime.catalog.sample(stem)
            image_path, geojson_path, generation_metadata = ensure_spatial_case(
                runtime, stem, args.seed
            )
            source_cells = load_cells(sample.geojson_path)
            predicted_cells = load_cells(geojson_path)
            for type_index, cell_type in enumerate(CELL_NAMES_WITH_TOTAL):
                metrics, matches, source_grid, predicted_grid = coordinate_metrics(
                    source_cells,
                    predicted_cells,
                    cell_type,
                    args.grid_size,
                    args.max_match_distance,
                    args.bootstrap,
                    args.seed + case_index * 100 + type_index,
                )
                per_image_rows.append(
                    {
                        "stem": stem,
                        "cell_type": cell_type,
                        **metrics,
                        "generated_image": str(image_path),
                        "cellvit_geojson": str(geojson_path),
                        "generation_seconds": generation_metadata.get("seconds", np.nan),
                    }
                )
                grids[cell_type].append((stem, source_grid, predicted_grid))
                for pair_index in range(len(matches.distances)):
                    match_rows.append(
                        {
                            "stem": stem,
                            "cell_type": cell_type,
                            "source_x": matches.source[pair_index, 0],
                            "source_y": matches.source[pair_index, 1],
                            "predicted_x": matches.predicted[pair_index, 0],
                            "predicted_y": matches.predicted[pair_index, 1],
                            "distance": matches.distances[pair_index],
                        }
                    )
            print(f"[{case_index}/{len(stems)}] completed {stem}", flush=True)
    finally:
        runtime.close()

    per_image = pd.DataFrame(per_image_rows)
    matches = pd.DataFrame(match_rows)
    per_image.to_csv(args.output_dir / "spatial_coordinate_per_image.csv", index=False)
    matches.to_csv(args.output_dir / "spatial_coordinate_matches.csv", index=False)

    summaries = []
    for type_index, cell_type in enumerate(CELL_NAMES_WITH_TOTAL):
        entries = grids[cell_type]
        source_values = np.concatenate([entry[1].ravel() for entry in entries])
        predicted_values = np.concatenate([entry[2].ravel() for entry in entries])
        groups = np.concatenate(
            [np.repeat(entry[0], entry[1].size) for entry in entries]
        )
        grid_result = spearman_with_bootstrap(
            source_values,
            predicted_values,
            bootstrap=args.bootstrap,
            seed=args.seed + type_index * 10,
            groups=groups,
        )
        type_matches = matches[matches["cell_type"] == cell_type] if len(matches) else matches
        x_result = spearman_with_bootstrap(
            type_matches["source_x"].to_numpy(float) if len(type_matches) else np.array([]),
            type_matches["predicted_x"].to_numpy(float) if len(type_matches) else np.array([]),
            bootstrap=args.bootstrap,
            seed=args.seed + type_index * 10 + 1,
        )
        y_result = spearman_with_bootstrap(
            type_matches["source_y"].to_numpy(float) if len(type_matches) else np.array([]),
            type_matches["predicted_y"].to_numpy(float) if len(type_matches) else np.array([]),
            bootstrap=args.bootstrap,
            seed=args.seed + type_index * 10 + 2,
        )
        subset = per_image[per_image["cell_type"] == cell_type]
        summaries.append(
            {
                "cell_type": cell_type,
                "images": len(subset),
                "pooled_grid_rho": grid_result.rho,
                "pooled_grid_p_value": grid_result.p_value,
                "pooled_grid_ci_low": grid_result.ci_low,
                "pooled_grid_ci_high": grid_result.ci_high,
                "median_per_image_grid_rho": finite_median(subset["grid_rho"]),
                "matched_pairs": len(type_matches),
                "pooled_matched_rho_x": x_result.rho,
                "pooled_matched_p_x": x_result.p_value,
                "pooled_matched_rho_y": y_result.rho,
                "pooled_matched_p_y": y_result.p_value,
                "median_match_distance": float(type_matches["distance"].median())
                if len(type_matches)
                else float("nan"),
                "median_matched_fraction": finite_median(subset["matched_fraction"]),
            }
        )
    summary = pd.DataFrame(summaries)
    summary["grid_p_fdr_bh"] = benjamini_hochberg(
        summary["pooled_grid_p_value"].tolist()
    )
    summary.to_csv(args.output_dir / "spatial_coordinate_spearman.csv", index=False)
    write_json(
        args.output_dir / "spatial_coordinate_manifest.json",
        {
            "experiment": "spatial_coordinate_fidelity",
            "num_images": len(stems),
            "primary_metric": "Spearman correlation of flattened class-specific centroid-count grids",
            "grid_size": args.grid_size,
            "secondary_metric": "Hungarian distance-matched x/y Spearman correlation",
            "secondary_metric_warning": "Matching is descriptive and may inflate coordinate correlation; use grid rho as primary.",
            "max_match_distance_pixels": args.max_match_distance,
            "input_coordinates": "exact centroids from source GeoJSON used to make conditioning maps",
            "generated_coordinates": "CellViT++ centroids on generated H&E",
            "guidance_hook": args.guidance_hook,
        },
    )
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
