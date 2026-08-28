#!/usr/bin/env python3
"""Merge Virchow2 and rotation results into the corrected paper table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from run_local_xai_rerun import build_table, markdown_table

MODEL_ORDER = ("resnet50", "ctranspath", "uni2h", "virchow2", "pathlupi_conch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-results-root", type=Path, required=True)
    parser.add_argument("--virchow-results-root", type=Path, required=True)
    parser.add_argument("--rotation-summary", type=Path, required=True)
    parser.add_argument("--pathlupi-rotation-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--primary-bag-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = args.base_results_root.resolve()
    virchow = args.virchow_results_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)

    summaries = pd.concat(
        [
            pd.read_csv(base / "experiment_summary_with_pathlupi.csv"),
            pd.read_csv(virchow / "experiment_summary.csv"),
            pd.read_csv(args.rotation_summary),
            pd.read_csv(args.pathlupi_rotation_summary),
        ],
        ignore_index=True,
    )
    summaries = summaries.drop_duplicates(
        ["model_id", "endpoint", "bag_size", "experiment"], keep="last"
    )
    performances = pd.concat(
        [
            pd.read_csv(base / "performance_with_pathlupi.csv"),
            pd.read_csv(virchow / "performance.csv"),
        ],
        ignore_index=True,
    ).drop_duplicates(["model_id", "endpoint", "bag_size"], keep="last")

    required_models = {"resnet50", "ctranspath", "uni2h", "virchow2"}
    rotation_models = set(
        summaries.loc[summaries["experiment"].eq("image_rotation"), "model_id"]
    )
    missing = sorted((required_models | {"pathlupi_conch"}) - rotation_models)
    if missing:
        raise RuntimeError(f"Rotation summaries are incomplete for: {missing}")

    table = build_table(
        summaries,
        performances,
        args.primary_bag_size,
        MODEL_ORDER,
    )
    summaries.to_csv(output / "experiment_summary_rotation_virchow2.csv", index=False)
    performances.to_csv(output / "performance_rotation_virchow2.csv", index=False)
    table.to_csv(output / "table4_rotation_virchow2.csv", index=False)
    (output / "table4_rotation_virchow2.md").write_text(
        markdown_table(table), encoding="utf-8"
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()


