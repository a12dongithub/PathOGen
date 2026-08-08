import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from cpathogen.utils.paths import MORPHOLOGY_DIR, MORPHOLOGY_STATS, TCGA_GEOJSON, TCGA_TILES


FEATURE_COLUMNS = (
    "area_mean", "area_var", "eccentricity_mean", "eccentricity_var",
    "solidity_mean", "solidity_var", "perimeter_mean", "perimeter_var",
    "grad_mean", "grad_var", "r_mean", "r_var", "g_mean", "g_var",
    "b_mean", "b_var",
)


def calculate_nuclei_features_single(img_path, geojson_path):
    """
    Computes aggregated features (Mean, Var) for all nuclei in a tile.
    Returns: dict of stats or None if error/empty.
    """
    try:
        # Load Image (BGR -> RGB)
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        if (h, w) != (512, 512):
            raise ValueError(f"Expected a 512 x 512 tile, found {w} x {h}")

        # Load GeoJSON
        with open(geojson_path, "r") as f:
            data = json.load(f)

        # Parse Polygons
        polygons = []
        if isinstance(data, list):
            features = data
        else:
            features = data.get("features", [])

        if not features:
            return None

        for feature in features:
            geom = feature.get("geometry", {})
            geom_type = geom.get("type")
            coords_list = geom.get("coordinates", [])

            if geom_type == "Polygon":
                # Outer ring is usually index 0
                if len(coords_list) > 0:
                    poly = np.array(coords_list[0], dtype=np.int32)
                    if len(poly) >= 3:
                        polygons.append(poly)
            elif geom_type == "MultiPolygon":
                for part in coords_list:
                    if len(part) > 0:
                        poly = np.array(part[0], dtype=np.int32)
                        if len(poly) >= 3:
                            polygons.append(poly)

        if len(polygons) == 0:
            return None

        # Precompute Gradient Magnitude
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)

        # Feature Lists
        areas = []
        perimeters = []
        solidities = []
        eccentricities = []
        grad_vals = []
        r_vals = []
        g_vals = []
        b_vals = []

        # Loop polygons
        for poly in polygons:
            # 1. Morphological Features
            area = cv2.contourArea(poly)
            perimeter = cv2.arcLength(poly, True)

            hull = cv2.convexHull(poly)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            # Eccentricity via Ellipse Fit
            if len(poly) >= 5:
                # fitEllipse returns (width, height) essentially. The larger is Major Axis.
                # Let's verify. Usually (MA, ma) order is not guaranteed.
                _, (axis_1, axis_2), _ = cv2.fitEllipse(poly)
                minor_axis = min(axis_1, axis_2)
                major_axis = max(axis_1, axis_2)

                if major_axis > 0:
                    # e = sqrt(1 - (b/a)^2)
                    eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
                else:
                    eccentricity = 0
            else:
                eccentricity = 0  # Cannot fit ellipse

            areas.append(area)
            perimeters.append(perimeter)
            solidities.append(solidity)
            eccentricities.append(eccentricity)

            # 2. Intensity Features (Mask)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [poly], -1, 1, -1)  # Draw filled polygon

            # Use mask to calculate mean intensity
            # cv2.mean returns (ch1, ch2, ch3, ch4)
            grad_mean = cv2.mean(grad_mag, mask=mask)[0]
            rgb_mean = cv2.mean(img, mask=mask)

            grad_vals.append(grad_mean)
            r_vals.append(rgb_mean[0])
            g_vals.append(rgb_mean[1])
            b_vals.append(rgb_mean[2])

        # Aggregate
        if len(areas) == 0:
            return None

        feats = {
            "area_mean": np.mean(areas),
            "area_var": np.var(areas),
            "eccentricity_mean": np.mean(eccentricities),
            "eccentricity_var": np.var(eccentricities),
            "solidity_mean": np.mean(solidities),
            "solidity_var": np.var(solidities),
            "perimeter_mean": np.mean(perimeters),
            "perimeter_var": np.var(perimeters),
            "grad_mean": np.mean(grad_vals),
            "grad_var": np.var(grad_vals),
            "r_mean": np.mean(r_vals),
            "r_var": np.var(r_vals),
            "g_mean": np.mean(g_vals),
            "g_var": np.var(g_vals),
            "b_mean": np.mean(b_vals),
            "b_var": np.var(b_vals),
        }

        return feats

    except Exception as e:  # noqa: BLE001 - report the failing tile to the orchestrator
        print(f"Error processing {img_path}: {e}")
        return None


def process_wrapper(args):
    """Wrapper for parallel processing to unpack args."""
    stem, img_path, geojson_path = args
    res = calculate_nuclei_features_single(img_path, geojson_path)
    return (stem, res)


def build_morphology_features(
    image_dir: str | Path,
    geojson_dir: str | Path,
    raw_output: str | Path,
    standardized_output: str | Path,
    scaler_output: str | Path,
    manifest_output: str | Path,
    n_jobs: int = 8,
    stems: Iterable[str] | None = None,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """Compute, standardize, and persist the 16-value tile condition table."""
    image_dir = Path(image_dir)
    geojson_dir = Path(geojson_dir)

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
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

    tasks = []
    for geojson_path in geojson_files:
        for suffix in (".png", ".jpg", ".jpeg"):
            image_path = image_dir / f"{geojson_path.stem}{suffix}"
            if image_path.exists():
                tasks.append((geojson_path.stem, image_path, geojson_path))
                break

    if not tasks:
        raise ValueError(
            f"No matching image/GeoJSON pairs found in {image_dir} and {geojson_dir}"
        )
    if requested_stems is not None:
        paired_stems = {task[0] for task in tasks}
        missing = sorted(requested_stems - paired_stems)
        if missing:
            raise ValueError(f"Missing tile files for stems: {missing[:5]}")

    print(f"Computing morphology/stain features for {len(tasks)} tiles...")
    results_list = Parallel(n_jobs=n_jobs)(
        delayed(process_wrapper)(task) for task in tqdm(tasks)
    )

    failed_stems = sorted(stem for stem, result in results_list if result is None)
    if failed_stems and not allow_empty:
        raise RuntimeError(
            f"Morphology extraction failed for {len(failed_stems)} tiles: "
            f"{failed_stems[:5]}"
        )

    valid_indices = [stem for stem, _ in results_list]
    valid_data = [
        result if result is not None else {column: np.nan for column in FEATURE_COLUMNS}
        for _, result in results_list
    ]
    df = pd.DataFrame(valid_data, index=valid_indices).sort_index()

    raw_output = Path(raw_output)
    standardized_output = Path(standardized_output)
    scaler_output = Path(scaler_output)
    manifest_output = Path(manifest_output)
    for output_path in (
        raw_output,
        standardized_output,
        scaler_output,
        manifest_output,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(raw_output)
    scaler = StandardScaler()
    normalized_values = scaler.fit_transform(df)
    normalized_df = pd.DataFrame(
        normalized_values,
        index=df.index,
        columns=df.columns,
    )
    normalized_df.to_parquet(standardized_output)
    dump(scaler, scaler_output)
    with manifest_output.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "feature_order": list(df.columns),
                "rows": len(df),
                "tile_stems": list(df.index),
                "raw_table": str(raw_output),
                "standardized_table": str(standardized_output),
                "scaler": str(scaler_output),
                "warning": "Scaler is valid only if input tiles are the sealed training split.",
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    print(f"Morphology conditions ready in {raw_output.parent}")
    return normalized_df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir", default=str(TCGA_TILES), help="Directory containing H&E tiles"
    )
    parser.add_argument(
        "--geojson-dir",
        default=str(TCGA_GEOJSON),
        help="Directory containing nucleus GeoJSON",
    )
    parser.add_argument(
        "--raw-output",
        default=str(MORPHOLOGY_DIR / "raw.parquet"),
    )
    parser.add_argument(
        "--output",
        default=str(MORPHOLOGY_STATS),
    )
    parser.add_argument(
        "--scaler-output",
        default=str(MORPHOLOGY_DIR / "scaler.joblib"),
    )
    parser.add_argument(
        "--manifest-output",
        default=str(MORPHOLOGY_DIR / "feature_manifest.json"),
    )
    parser.add_argument("--n-jobs", "--n_jobs", dest="n_jobs", type=int, default=8)
    args = parser.parse_args(argv)
    build_morphology_features(
        args.image_dir,
        args.geojson_dir,
        args.raw_output,
        args.output,
        args.scaler_output,
        args.manifest_output,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
