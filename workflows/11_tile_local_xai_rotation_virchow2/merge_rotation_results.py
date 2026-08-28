#!/usr/bin/env python3
"""Merge rotation results for a requested subset of endpoint models."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from run_local_xai_rerun import build_table, markdown_table

DEFAULT_MODELS = ("resnet50", "ctranspath", "uni2h", "pathlupi_conch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-results-root", type=Path, required=True)
    parser.add_argument("--rotation-summary", type=Path, nargs="+", required=True)
    parser.add_argument("--pathlupi-rotation-summary", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--primary-bag-size", type=int, default=16)
    parser.add_argument("--output-stem", default="table4_rotation_without_virchow")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = args.base_results_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)

    summary_frames = [
        pd.read_csv(base / "experiment_summary_with_pathlupi.csv"),
        *(pd.read_csv(path) for path in args.rotation_summary),
    ]
    if args.pathlupi_rotation_summary is not None:
        summary_frames.append(pd.read_csv(args.pathlupi_rotation_summary))
    summaries = pd.concat(summary_frames, ignore_index=True).drop_duplicates(
        ["model_id", "endpoint", "bag_size", "experiment"], keep="last"
    )
    performances = pd.read_csv(base / "performance_with_pathlupi.csv")
    performances = performances.drop_duplicates(
        ["model_id", "endpoint", "bag_size"], keep="last"
    )

    requested = set(args.models)
    rotation_models = set(
        summaries.loc[summaries["experiment"].eq("image_rotation"), "model_id"]
    )
    missing = sorted(requested - rotation_models)
    if missing:
        raise RuntimeError(f"Rotation summaries are incomplete for: {missing}")

    table = build_table(
        summaries,
        performances,
        args.primary_bag_size,
        args.models,
    )
    summaries.to_csv(output / "experiment_summary_with_rotation.csv", index=False)
    performances.to_csv(output / "performance_with_rotation.csv", index=False)
    table.to_csv(output / f"{args.output_stem}.csv", index=False)
    (output / f"{args.output_stem}.md").write_text(
        markdown_table(table), encoding="utf-8"
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
