#!/usr/bin/env python3
"""Create an on-the-fly 0/90/180/270-degree rotation intervention manifest."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

ROTATIONS = (
    ("rotation_0", 0),
    ("rotation_90", 4),
    ("rotation_180", 3),
    ("rotation_270", 5),
)

SOURCE_REFERENCES = (
    ("nuclear_shape_irregularity", "baseline"),
    ("nuclear_enlargement", "baseline"),
    ("stain_brightness", "baseline"),
    ("peritumoral_immune_ring_diameter40px", "peritumoral_ring_plus_80"),
    ("tumor_immune_separation_diameter40px", "tumor_immune_maximal_mixing"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterfactual-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-manifest", type=Path, required=True)
    parser.add_argument(
        "--local-image-cache-dir",
        type=Path,
        help="Optionally copy each selected source image here once for faster I/O.",
    )
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.counterfactual_root.expanduser().resolve()
    source_manifest = root / "organized_bucket_images.csv"
    frame = pd.read_csv(source_manifest)
    stem_column = "stem" if "stem" in frame else "source_tile_id"
    allowed_tiles = set(
        pd.read_csv(args.tile_manifest)["tile_id"].astype(str)
    )
    candidates = []
    seen: set[str] = set()
    for source_experiment, reference_condition in SOURCE_REFERENCES:
        subset = frame[
            frame["experiment"].eq(source_experiment)
            & frame["condition"].eq(reference_condition)
        ].copy()
        subset["stem_resolved"] = subset[stem_column].astype(str)
        subset = subset.sort_values("stem_resolved", kind="stable")
        for row in subset.to_dict("records"):
            stem = str(row["stem_resolved"])
            if stem not in allowed_tiles or stem in seen:
                continue
            seen.add(stem)
            row["source_experiment_resolved"] = source_experiment
            row["source_condition_resolved"] = reference_condition
            candidates.append(row)
    candidates = pd.DataFrame(candidates)
    if len(candidates) < args.num_images:
        raise RuntimeError(
            f"Only {len(candidates)} baseline images are available; "
            f"requested {args.num_images}"
        )
    selected = candidates.head(args.num_images)
    local_cache = (
        args.local_image_cache_dir.expanduser().resolve()
        if args.local_image_cache_dir
        else None
    )
    if local_cache is not None:
        local_cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in selected.itertuples(index=False):
        stem = str(source.stem_resolved)
        source_experiment = str(source.source_experiment_resolved)
        source_condition = str(source.source_condition_resolved)
        image = root / source_experiment / stem / f"{source_condition}.png"
        if not image.is_file():
            raise FileNotFoundError(image)
        if local_cache is not None:
            cached_image = local_cache / f"{stem}.png"
            if not cached_image.is_file() or cached_image.stat().st_size != image.stat().st_size:
                shutil.copy2(image, cached_image)
            image = cached_image
        for condition, augmentation_code in ROTATIONS:
            rows.append(
                {
                    "stem": stem,
                    "experiment": "image_rotation",
                    "condition": condition,
                    "local_path": str(image),
                    "augmentation_code": augmentation_code,
                    "seed": args.seed,
                    "source_experiment": source_experiment,
                    "source_condition": source_condition,
                }
            )
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output / "images.csv", index=False)
    print(
        f"Wrote {len(manifest)} manifest rows for "
        f"{manifest['stem'].nunique()} source tiles to {output / 'images.csv'}"
    )


if __name__ == "__main__":
    main()
