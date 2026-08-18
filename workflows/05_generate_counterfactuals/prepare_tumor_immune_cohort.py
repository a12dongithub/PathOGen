#!/usr/bin/env python3
"""Create an exact-centroid tumor-immune cohort from reranked candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from cpathogen.counterfactuals import (
    cell_centroids_by_class_from_geojson,
    render_centroid_channel,
)
from cpathogen.counterfactuals.conditions import MORPHOLOGY_FEATURE_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a stable tumor/immune-positive cohort while retaining the "
            "diffusion seed supplied by selected_candidates.csv."
        )
    )
    parser.add_argument("--selected-candidates", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-candidates", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rendered_uint8(centroids: np.ndarray) -> np.ndarray:
    return np.rint(render_centroid_channel(centroids).numpy() * 255.0).astype(
        np.uint8
    )


def _validate_control_channel(
    spatial_map: np.ndarray,
    centroids: np.ndarray,
    *,
    channel_index: int,
    stem: str,
    classification: str,
) -> None:
    rendered = _rendered_uint8(centroids)
    stored = np.asarray(spatial_map[:, :, channel_index], dtype=np.uint8)
    if not np.array_equal(rendered, stored):
        maximum_delta = int(
            np.abs(rendered.astype(np.int16) - stored.astype(np.int16)).max()
        )
        raise ValueError(
            f"{classification} centroids do not reproduce {stem}.npz "
            f"(maximum uint8 delta={maximum_delta})"
        )


def main() -> None:
    args = parse_args()
    if args.num_candidates < 1:
        raise ValueError("--num-candidates must be positive")

    selected_path = args.selected_candidates.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if not selected_path.is_file():
        raise FileNotFoundError(selected_path)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; use --overwrite")

    table = pd.read_csv(selected_path)
    required = {"stem", "seed"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Selected-candidate CSV is missing columns: {missing}")
    morphology_path = data_root / "morphology_stats.parquet"
    if not morphology_path.is_file():
        morphology_path = data_root / "morphology" / "standardized.parquet"
    if not morphology_path.is_file():
        raise FileNotFoundError("Could not find the standardized morphology table")
    morphology = pd.read_parquet(morphology_path)
    morphology.index = morphology.index.map(str)
    missing_features = sorted(set(MORPHOLOGY_FEATURE_NAMES).difference(morphology))
    if missing_features:
        raise ValueError(f"Morphology table is missing columns: {missing_features}")

    accepted: list[dict[str, object]] = []
    exclusions: Counter[str] = Counter()
    seen_stems: set[str] = set()
    for source_row_index, row in table.iterrows():
        stem = str(row["stem"])
        if stem in seen_stems:
            exclusions["duplicate_stem"] += 1
            continue
        seen_stems.add(stem)
        map_path = data_root / "spatial_maps" / f"{stem}.npz"
        geojson_path = data_root / "geojsons" / f"{stem}.geojson"
        if stem not in morphology.index:
            exclusions["missing_morphology"] += 1
            continue
        if not map_path.is_file():
            exclusions["missing_spatial_map"] += 1
            continue
        if not geojson_path.is_file():
            exclusions["missing_geojson"] += 1
            continue

        by_class = cell_centroids_by_class_from_geojson(geojson_path)
        tumor = by_class.get("Neoplastic", np.empty((0, 2), dtype=np.int16))
        inflammatory = by_class.get(
            "Inflammatory", np.empty((0, 2), dtype=np.int16)
        )
        if len(tumor) == 0:
            exclusions["zero_neoplastic_centroids"] += 1
            continue
        if len(inflammatory) == 0:
            exclusions["zero_inflammatory_centroids"] += 1
            continue

        with np.load(map_path, allow_pickle=False) as archive:
            if "map" not in archive:
                raise ValueError(f"Expected key 'map' in {map_path}")
            spatial_map = np.asarray(archive["map"])
        if spatial_map.shape != (512, 512, 5):
            raise ValueError(f"Unexpected spatial-map shape for {stem}: {spatial_map.shape}")
        _validate_control_channel(
            spatial_map,
            tumor,
            channel_index=0,
            stem=stem,
            classification="Neoplastic",
        )
        _validate_control_channel(
            spatial_map,
            inflammatory,
            channel_index=1,
            stem=stem,
            classification="Inflammatory",
        )

        record: dict[str, object] = {
            "candidate_id": f"candidate_{len(accepted):04d}",
            "stem": stem,
            "seed": int(row["seed"]),
            "source_row_index": int(source_row_index),
            "neoplastic_centroid_count": len(tumor),
            "inflammatory_centroid_count": len(inflammatory),
        }
        for name in (
            "candidate_order",
            "seed_index",
            "config_id",
            "green_sd",
            "controlnet_strength",
            "denoising_steps",
            "spatial_score",
        ):
            if name in table.columns and not pd.isna(row[name]):
                record[name] = row[name]
        accepted.append(record)
        if len(accepted) == args.num_candidates:
            break

    if len(accepted) != args.num_candidates:
        raise RuntimeError(
            f"Requested {args.num_candidates} eligible cases, found {len(accepted)}; "
            f"exclusions={dict(exclusions)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    pd.DataFrame(accepted).to_csv(temporary, index=False)
    temporary.replace(output)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_candidates_source_name": selected_path.name,
        "selected_candidates_sha256": _sha256(selected_path),
        "data_root_layout": data_root.name,
        "morphology_table": str(morphology_path.relative_to(data_root)),
        "selection_rule": (
            "CSV order; unique stem; aligned morphology and spatial map; source "
            "GeoJSON contains >=1 Neoplastic and >=1 Inflammatory centroid; "
            "both rendered centroid channels exactly reproduce stored uint8 controls"
        ),
        "requested_candidate_count": args.num_candidates,
        "selected_candidate_count": len(accepted),
        "rows_scanned_through": int(accepted[-1]["source_row_index"]) + 1,
        "exclusions_before_completion": dict(sorted(exclusions.items())),
        "cohort_csv": output.name,
        "cohort_csv_sha256": _sha256(output),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(accepted)} candidates to {output}")
    print(json.dumps(manifest["exclusions_before_completion"], sort_keys=True))


if __name__ == "__main__":
    main()
