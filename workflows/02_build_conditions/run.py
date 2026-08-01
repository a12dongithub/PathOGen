"""Workflow 02: build model conditions from matched tiles and GeoJSON files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cpathogen.preprocessing.metadata import build_metadata
from cpathogen.preprocessing.morphology_features import build_morphology_features
from cpathogen.preprocessing.spatial_maps import build_spatial_maps
from cpathogen.utils.paths import CONDITIONS_ROOT, TCGA_GEOJSON, TCGA_TILES


def _tile_stems(directory: Path) -> set[str]:
    stems: set[str] = set()
    for extension in ("*.png", "*.jpg", "*.jpeg"):
        stems.update(path.stem for path in directory.glob(extension))
    return stems


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build five-channel spatial maps, 16-value morphology/stain "
            "conditions, a fitted scaler, and ImageFolder metadata."
        )
    )
    parser.add_argument("--tiles-dir", default=str(TCGA_TILES))
    parser.add_argument("--geojson-dir", default=str(TCGA_GEOJSON))
    parser.add_argument("--output-dir", default=str(CONDITIONS_ROOT))
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument(
        "--allow-unmatched",
        action="store_true",
        help="Process only the intersection when either input directory has unmatched stems.",
    )
    parser.add_argument(
        "--overwrite-spatial-maps",
        action="store_true",
        help="Regenerate spatial-map NPZ files that already exist.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    tiles_dir = Path(args.tiles_dir)
    geojson_dir = Path(args.geojson_dir)
    output_dir = Path(args.output_dir)

    if not tiles_dir.is_dir():
        raise FileNotFoundError(f"Tile directory does not exist: {tiles_dir}")
    if not geojson_dir.is_dir():
        raise FileNotFoundError(f"GeoJSON directory does not exist: {geojson_dir}")

    tile_stems = _tile_stems(tiles_dir)
    geojson_stems = {path.stem for path in geojson_dir.glob("*.geojson")}
    matched_stems = sorted(tile_stems & geojson_stems)
    tiles_without_geojson = sorted(tile_stems - geojson_stems)
    geojson_without_tiles = sorted(geojson_stems - tile_stems)

    if not matched_stems:
        raise ValueError("No matching tile/GeoJSON stems were found")
    if (tiles_without_geojson or geojson_without_tiles) and not args.allow_unmatched:
        raise ValueError(
            "Input directories must have one-to-one matching stems. "
            f"Tiles without GeoJSON: {tiles_without_geojson[:5]}; "
            f"GeoJSON without tiles: {geojson_without_tiles[:5]}. "
            "Use --allow-unmatched to process only the intersection."
        )

    spatial_dir = output_dir / "spatial_maps"
    morphology_dir = output_dir / "morphology"
    metadata_path = output_dir / "metadata.jsonl"

    print(f"Building conditions for {len(matched_stems)} matched tiles")
    build_spatial_maps(
        geojson_dir,
        spatial_dir,
        n_jobs=args.n_jobs,
        stems=matched_stems,
        overwrite=args.overwrite_spatial_maps,
    )
    morphology_table = build_morphology_features(
        tiles_dir,
        geojson_dir,
        morphology_dir / "raw.parquet",
        morphology_dir / "standardized.parquet",
        morphology_dir / "scaler.joblib",
        morphology_dir / "feature_manifest.json",
        n_jobs=args.n_jobs,
        stems=matched_stems,
    )
    metadata_images = build_metadata(
        tiles_dir,
        metadata_path,
        stems=matched_stems,
    )

    expected_stems = set(matched_stems)
    map_stems = {path.stem for path in spatial_dir.glob("*.npz")}
    if not expected_stems.issubset(map_stems):
        raise RuntimeError("One or more requested spatial maps are missing")
    for stem in matched_stems:
        with np.load(spatial_dir / f"{stem}.npz", allow_pickle=False) as payload:
            if "map" not in payload:
                raise RuntimeError(f"Spatial map has no 'map' key: {stem}")
            spatial_map = payload["map"]
        if spatial_map.shape != (512, 512, 5) or spatial_map.dtype != np.uint8:
            raise RuntimeError(
                f"Invalid spatial map contract for {stem}: "
                f"shape={spatial_map.shape}, dtype={spatial_map.dtype}"
            )
    if set(morphology_table.index) != expected_stems:
        raise RuntimeError("Morphology-table rows do not match the requested stems")
    if morphology_table.shape != (len(expected_stems), 16):
        raise RuntimeError(
            f"Expected a {len(expected_stems)} x 16 morphology table, "
            f"found {morphology_table.shape}"
        )
    if not np.isfinite(morphology_table.to_numpy()).all():
        raise RuntimeError("Morphology table contains NaN or infinite values")
    if {path.stem for path in metadata_images} != expected_stems:
        raise RuntimeError("Metadata rows do not match the requested stems")

    print("Condition build complete:")
    print(f"  spatial maps: {spatial_dir}")
    print(f"  morphology:   {morphology_dir}")
    print(f"  metadata:     {metadata_path}")


if __name__ == "__main__":
    main()
