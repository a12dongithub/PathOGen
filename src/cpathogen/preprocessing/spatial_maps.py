import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from cpathogen.utils.paths import SPATIAL_MAPS, TCGA_GEOJSON

TYPE_MAP = {
    "Neoplastic": 0,
    "Inflammatory": 1,
    "Connective": 2,
    "Dead": 3,
    "Epithelial": 4,
}
NUM_CLASSES = 5
IMG_SIZE = 512
SIGMA = 3.0


def process_single_geojson(geojson_path, out_dir, overwrite=False):
    try:
        geojson_path = Path(geojson_path)
        out_dir = Path(out_dir)
        stem = geojson_path.stem
        out_file = out_dir / f"{stem}.npz"

        # Skip if exists
        if out_file.exists() and not overwrite:
            return stem

        with geojson_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        features = data if isinstance(data, list) else data.get("features", [])

        # Initialize map: (H, W, K)
        spatial_map = np.zeros((IMG_SIZE, IMG_SIZE, NUM_CLASSES), dtype=np.float32)

        for feature in features:
            props = feature.get("properties", {})
            cls_obj = props.get("classification", {})
            c_name = cls_obj.get("name", "Unknown")

            c_idx = TYPE_MAP.get(c_name, -1)
            if c_idx == -1:
                # If unknown type is encountered, we skip or add a map? We ignore unknown.
                continue

            geom = feature.get("geometry", {})
            geom_type = geom.get("type")
            coords_list = geom.get("coordinates", [])

            polygons = []
            if geom_type == "Polygon" and len(coords_list) > 0:
                polygons.append(np.array(coords_list[0], dtype=np.int32))
            elif geom_type == "MultiPolygon":
                for part in coords_list:
                    if len(part) > 0:
                        polygons.append(np.array(part[0], dtype=np.int32))

            for poly in polygons:
                if len(poly) > 0:
                    # Calculate centroid
                    mean_x = np.mean(poly[:, 0])
                    mean_y = np.mean(poly[:, 1])
                    cx, cy = round(mean_x), round(mean_y)

                    if 0 <= cx < IMG_SIZE and 0 <= cy < IMG_SIZE:
                        spatial_map[cy, cx, c_idx] += 1.0  # Accumulate

        # Apply Gaussian filter per channel
        for i in range(NUM_CLASSES):
            if spatial_map[:, :, i].max() > 0:
                spatial_map[:, :, i] = gaussian_filter(
                    spatial_map[:, :, i], sigma=SIGMA
                )
                # Normalize peak to 1.0
                c_max = spatial_map[:, :, i].max()
                if c_max > 0:
                    spatial_map[:, :, i] = spatial_map[:, :, i] / c_max

        # Clip, scale to 0-255, and convert to uint8 to save massive space (512x512x5 float16 = 2.5MB, uint8 compressed = <50KB)
        spatial_map = (np.clip(spatial_map, 0, 1) * 255.0).astype(np.uint8)

        # Save compressed (often 90% sparse, so size drops drastically)
        np.savez_compressed(out_file, map=spatial_map)
        return stem

    except Exception as e:  # noqa: BLE001 - report the failing GeoJSON to the orchestrator
        print(f"Error processing {geojson_path}: {e}")
        return None


def build_spatial_maps(
    geojson_dir: str | Path,
    output_dir: str | Path,
    n_jobs: int = 8,
    stems: Iterable[str] | None = None,
    overwrite: bool = False,
) -> list[str]:
    """Create one five-channel spatial-map NPZ for each selected GeoJSON."""
    geojson_dir = Path(geojson_dir)
    out_dir = Path(output_dir)

    if not geojson_dir.is_dir():
        raise FileNotFoundError(f"GeoJSON directory does not exist: {geojson_dir}")

    requested_stems = set(stems) if stems is not None else None
    geojson_files = sorted(geojson_dir.glob("*.geojson"))
    if requested_stems is not None:
        geojson_files = [path for path in geojson_files if path.stem in requested_stems]
        found_stems = {path.stem for path in geojson_files}
        missing = sorted(requested_stems - found_stems)
        if missing:
            raise ValueError(f"Missing GeoJSON files for stems: {missing[:5]}")

    if not geojson_files:
        raise ValueError(f"No GeoJSON files found in {geojson_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(geojson_files)} spatial maps with {n_jobs} job(s)...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_geojson)(path, out_dir, overwrite)
        for path in tqdm(geojson_files)
    )

    valid = sorted(result for result in results if result is not None)
    if len(valid) != len(geojson_files):
        raise RuntimeError(
            f"Generated {len(valid)} of {len(geojson_files)} requested spatial maps"
        )
    print(f"Spatial maps ready in {out_dir}")
    return valid


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geojson-dir",
        default=str(TCGA_GEOJSON),
        help="Directory containing per-tile nucleus GeoJSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SPATIAL_MAPS),
        help="Directory for generated five-channel NPZ maps.",
    )
    parser.add_argument("--n-jobs", "--n_jobs", dest="n_jobs", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    build_spatial_maps(
        args.geojson_dir,
        args.output_dir,
        n_jobs=args.n_jobs,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
