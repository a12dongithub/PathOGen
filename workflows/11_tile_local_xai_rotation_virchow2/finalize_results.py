#!/usr/bin/env python3
"""Merge corrected endpoint results and emit deterministic quality checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    root = parse_args().results_root.resolve()
    base_summary = pd.read_csv(root / "experiment_summary.csv")
    pathlupi_summary = pd.read_csv(root / "pathlupi_experiment_summary.csv")
    summary = pd.concat([base_summary, pathlupi_summary], ignore_index=True)
    summary.to_csv(root / "experiment_summary_with_pathlupi.csv", index=False)

    base_detail = pd.read_parquet(root / "pair_metrics.parquet")
    pathlupi_detail = pd.read_parquet(root / "pathlupi_pair_metrics.parquet")
    detail = pd.concat([base_detail, pathlupi_detail], ignore_index=True)
    detail.to_parquet(root / "pair_metrics_with_pathlupi.parquet", index=False)

    base_performance = pd.read_csv(root / "performance.csv")
    pathlupi_performance = pd.read_csv(root / "pathlupi_performance.csv")
    performance = pd.concat([base_performance, pathlupi_performance], ignore_index=True)
    performance.to_csv(root / "performance_with_pathlupi.csv", index=False)

    table = pd.read_csv(root / "table4_revised_with_pathlupi.csv")
    duplicate_columns = [
        "model_id",
        "endpoint",
        "bag_size",
        "experiment",
        "source_tile_id",
        "target_condition",
    ]
    duplicates = int(detail.duplicated(duplicate_columns).sum())
    finite_tvd = bool(np.isfinite(detail["tvd"]).all())
    finite_flip = bool(np.isfinite(detail["flip"]).all())
    tvd_in_range = bool(detail["tvd"].between(0.0, 1.0).all())
    flip_in_range = bool(detail["flip"].between(0.0, 1.0).all())

    bnr_checks = []
    for (model_id, endpoint, bag_size), group in summary.groupby(
        ["model_id", "endpoint", "bag_size"], dropna=False
    ):
        stain = group.loc[group["family"].eq("nuisance"), "mean_tvd"]
        biological = group.loc[group["family"].eq("biological"), "mean_tvd"]
        if len(stain) == 1 and len(biological) == 4 and stain.iloc[0] > 0:
            bnr_checks.append(
                {
                    "model_id": model_id,
                    "endpoint": endpoint,
                    "bag_size": None if pd.isna(bag_size) else int(bag_size),
                    "recomputed_bnr": float(biological.mean() / stain.iloc[0]),
                }
            )

    qc = {
        "status": "pass"
        if finite_tvd and finite_flip and tvd_in_range and flip_in_range and duplicates == 0
        else "fail",
        "table_rows": len(table),
        "summary_rows": len(summary),
        "pair_rows": len(detail),
        "performance_rows": len(performance),
        "duplicate_pair_keys": duplicates,
        "finite_tvd": finite_tvd,
        "finite_flip": finite_flip,
        "tvd_in_0_1": tvd_in_range,
        "flip_in_0_1": flip_in_range,
        "tvd_min": float(detail["tvd"].min()),
        "tvd_max": float(detail["tvd"].max()),
        "experiment_panel_counts": json.loads(
            summary[
                ["model", "endpoint", "bag_size", "display_experiment", "tiles", "patients"]
            ].to_json(orient="records")
        ),
        "bnr_recomputed": bnr_checks,
    }
    (root / "quality_control.json").write_text(
        json.dumps(qc, indent=2), encoding="utf-8"
    )
    if qc["status"] != "pass":
        raise SystemExit(json.dumps(qc, indent=2))
    print(json.dumps({key: value for key, value in qc.items() if key != "experiment_panel_counts"}, indent=2))


if __name__ == "__main__":
    main()


