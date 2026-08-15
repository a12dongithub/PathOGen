#!/usr/bin/env python3
"""Build a compact data archive for inflammatory-centroid interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from cpathogen.counterfactuals.centroids import (
    inflammatory_centroids_from_geojson,
    render_centroid_channel,
)
from cpathogen.counterfactuals.conditions import (
    CELL_TYPE_NAMES,
    MORPHOLOGY_FEATURE_NAMES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selected-candidates",
        type=Path,
        default=REPOSITORY_ROOT / "data/selected_candidates.csv",
    )
    parser.add_argument(
        "--spatial-maps-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data/spatial_maps",
    )
    parser.add_argument(
        "--morphology-table",
        type=Path,
        default=REPOSITORY_ROOT / "data/morphology_stats.parquet",
    )
    parser.add_argument("--geojson-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=300)
    parser.add_argument("--minimum-inflammatory-centroids", type=int, default=10)
    parser.add_argument(
        "--exclude-candidates",
        type=Path,
        help="CSV whose stems must be excluded from this artifact.",
    )
    parser.add_argument(
        "--candidate-id-offset",
        type=int,
        default=0,
        help="Integer added to generated candidate IDs.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_zip(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.name != ".DS_Store":
                archive.write(path, path.relative_to(source))
    temporary.replace(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{_sha256(destination)}  {destination.name}\n", encoding="utf-8"
    )


def _select_candidates(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, float | int | str]]:
    table = pd.read_csv(args.selected_candidates)
    required = {"stem", "seed", "green_sd", "spatial_score"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Selected-candidate CSV is missing columns: {missing}")
    table = table.loc[table["green_sd"] == 0.0].copy()
    table["stem"] = table["stem"].astype(str)
    centroid_controls: dict[str, np.ndarray] = {}
    counts = []
    for stem in table["stem"]:
        path = args.geojson_dir / f"{stem}.geojson"
        if not path.is_file():
            raise FileNotFoundError(f"GeoJSON not found: {path}")
        centroids = inflammatory_centroids_from_geojson(path)
        centroid_controls[stem] = centroids
        counts.append(len(centroids))
    table["inflammatory_centroid_count"] = counts
    positive_counts = np.asarray([count for count in counts if count > 0], dtype=float)
    if len(positive_counts) < 2:
        raise ValueError("At least two positive inflammatory counts are required")
    reference_stats: dict[str, float | int | str] = {
        "transform": "sqrt",
        "sqrt_count_sd": float(np.sqrt(positive_counts).std(ddof=1)),
        "reference_count": len(positive_counts),
        "reference_population": "neutral-green selected candidates with count > 0",
    }
    table = table.loc[
        table["inflammatory_centroid_count"] >= args.minimum_inflammatory_centroids
    ].sort_values(["spatial_score", "stem"], ascending=[False, True], kind="stable")
    if args.exclude_candidates:
        excluded = pd.read_csv(args.exclude_candidates)
        if "stem" not in excluded.columns:
            raise ValueError("Excluded-candidate CSV is missing the stem column")
        excluded_stems = set(excluded["stem"].dropna().astype(str))
        table = table.loc[~table["stem"].isin(excluded_stems)]
    table = table.head(args.candidate_count).copy()
    if len(table) != args.candidate_count:
        raise ValueError(
            f"Needed {args.candidate_count} eligible candidates, found {len(table)}"
        )
    if not table["stem"].is_unique:
        raise ValueError("Selected candidate stems must be unique")
    table.insert(
        0,
        "candidate_id",
        [
            f"candidate_{index:04d}"
            for index in range(
                args.candidate_id_offset,
                args.candidate_id_offset + args.candidate_count,
            )
        ],
    )
    return table, centroid_controls, reference_stats


def _validate_rendered_channel(map_path: Path, centroids: np.ndarray) -> None:
    with np.load(map_path, allow_pickle=False) as archive:
        spatial_map = np.asarray(archive["map"], dtype=np.uint8)
    if spatial_map.shape != (512, 512, len(CELL_TYPE_NAMES)):
        raise ValueError(f"Unexpected spatial-map shape: {map_path}")
    rendered = np.rint(render_centroid_channel(centroids).numpy() * 255.0).astype(
        np.uint8
    )
    if not np.array_equal(rendered, spatial_map[:, :, 1]):
        difference = int(np.abs(rendered.astype(int) - spatial_map[:, :, 1]).max())
        raise ValueError(
            f"Centroid rendering does not reproduce {map_path.name}; max delta={difference}"
        )


def main() -> None:
    args = parse_args()
    if args.candidate_count < 1:
        raise ValueError("--candidate-count must be positive")
    if args.minimum_inflammatory_centroids < 1:
        raise ValueError("At least one original inflammatory centroid is required")
    if args.candidate_id_offset < 0:
        raise ValueError("--candidate-id-offset must be non-negative")
    selected, centroid_controls, reference_stats = _select_candidates(args)

    morphology = pd.read_parquet(args.morphology_table)
    morphology.index = morphology.index.map(str)
    missing_morphology = sorted(set(selected["stem"]).difference(morphology.index))
    if missing_morphology:
        raise KeyError(
            f"Selected stems lack morphology rows: {missing_morphology[:10]}"
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / (
        f"cpathogen_inflammatory_centroid_density_{len(selected)}_data.zip"
    )
    if archive_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing artifact: {archive_path}"
        )

    with tempfile.TemporaryDirectory(prefix="cpathogen-centroids-") as temporary:
        staging = Path(temporary) / "data_bundle" / "data"
        maps_output = staging / "spatial_maps"
        centroids_output = staging / "cell_centroids"
        maps_output.mkdir(parents=True)
        centroids_output.mkdir(parents=True)
        selected.to_csv(staging / "selected_candidates.csv", index=False)
        reference_stats_path = centroids_output / "reference_stats.json"
        reference_stats_path.write_text(
            json.dumps(reference_stats, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        morphology.loc[selected["stem"], list(MORPHOLOGY_FEATURE_NAMES)].to_parquet(
            staging / "morphology_stats.parquet"
        )

        file_hashes: dict[str, dict[str, str]] = {
            "spatial_maps": {},
            "cell_centroids": {},
        }
        file_hashes["cell_centroids"][reference_stats_path.name] = _sha256(
            reference_stats_path
        )
        for stem in selected["stem"]:
            source_map = args.spatial_maps_dir / f"{stem}.npz"
            if not source_map.is_file():
                raise FileNotFoundError(f"Spatial map not found: {source_map}")
            centroids = centroid_controls[stem]
            _validate_rendered_channel(source_map, centroids)
            destination_map = maps_output / source_map.name
            shutil.copy2(source_map, destination_map)
            centroid_path = centroids_output / f"{stem}.npz"
            np.savez_compressed(centroid_path, inflammatory_xy=centroids)
            file_hashes["spatial_maps"][source_map.name] = _sha256(destination_map)
            file_hashes["cell_centroids"][centroid_path.name] = _sha256(centroid_path)

        manifest = {
            "schema_version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "selection": {
                "source_sha256": _sha256(args.selected_candidates),
                "excluded_candidates_sha256": (
                    _sha256(args.exclude_candidates)
                    if args.exclude_candidates
                    else None
                ),
                "rule": (
                    "green_sd == 0; inflammatory centroid count >= "
                    f"{args.minimum_inflammatory_centroids}; spatial_score descending; "
                    "stem ascending tie-break; excluded stems removed"
                ),
                "candidate_count": len(selected),
                "candidate_id_offset": args.candidate_id_offset,
            },
            "intervention_levels_sd": [0.0, 0.5, 1.0, 1.5],
            "centroid_count_reference": reference_stats,
            "spatial_map": {
                "archive_key": "map",
                "layout": "HWC",
                "channel_order": list(CELL_TYPE_NAMES),
                "render": "centroid impulses; Gaussian sigma=3; per-channel peak normalization; uint8",
            },
            "centroid_control": {
                "archive_key": "inflammatory_xy",
                "coordinate_order": ["x", "y"],
                "source": "original nucleus GeoJSON polygon centers",
            },
            "morphology_feature_order": list(MORPHOLOGY_FEATURE_NAMES),
            "files": file_hashes,
        }
        (staging / "dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_zip(staging.parent, archive_path)

    print(archive_path)


if __name__ == "__main__":
    main()
