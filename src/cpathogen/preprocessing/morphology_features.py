
import argparse
import os
import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed, dump

from cpathogen.utils.paths import MORPHOLOGY_FEATURES, TCGA_GEOJSON, TCGA_TILES

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
        
        # Load GeoJSON
        with open(geojson_path, 'r') as f:
            data = json.load(f)
            
        # Parse Polygons
        polygons = []
        if isinstance(data, list):
            features = data
        else:
            features = data.get('features', [])
            
        if not features:
            return None
            
        for feature in features:
            geom = feature.get('geometry', {})
            geom_type = geom.get('type')
            coords_list = geom.get('coordinates', [])
            
            if geom_type == 'Polygon':
                # Outer ring is usually index 0
                if len(coords_list) > 0:
                    poly = np.array(coords_list[0], dtype=np.int32)
                    if len(poly) >= 3:
                        polygons.append(poly)
            elif geom_type == 'MultiPolygon':
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
                (cx, cy), (ma, MA), angle = cv2.fitEllipse(poly)
                d1, d2 = ma, MA
                ma = min(d1, d2)
                MA = max(d1, d2)
                
                if MA > 0:
                    # e = sqrt(1 - (b/a)^2)
                    eccentricity = np.sqrt(1 - (ma / MA)**2)
                else:
                    eccentricity = 0
            else:
                eccentricity = 0 # Cannot fit ellipse
                
            areas.append(area)
            perimeters.append(perimeter)
            solidities.append(solidity)
            eccentricities.append(eccentricity)
            
            # 2. Intensity Features (Mask)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [poly], -1, 1, -1) # Draw filled polygon
            
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
            'area_mean': np.mean(areas),
            'area_var': np.var(areas),
            'eccentricity_mean': np.mean(eccentricities),
            'eccentricity_var': np.var(eccentricities),
            'solidity_mean': np.mean(solidities),
            'solidity_var': np.var(solidities),
            'perimeter_mean': np.mean(perimeters),
            'perimeter_var': np.var(perimeters),
            'grad_mean': np.mean(grad_vals),
            'grad_var': np.var(grad_vals),
            'r_mean': np.mean(r_vals),
            'r_var': np.var(r_vals),
            'g_mean': np.mean(g_vals),
            'g_var': np.var(g_vals),
            'b_mean': np.mean(b_vals),
            'b_var': np.var(b_vals)
        }
        
        return feats
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def process_wrapper(args):
    """Wrapper for parallel processing to unpack args."""
    stem, img_path, geojson_path = args
    res = calculate_nuclei_features_single(img_path, geojson_path)
    return (stem, res)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default=str(TCGA_TILES), help="Directory containing H&E tiles")
    parser.add_argument("--geojson-dir", default=str(TCGA_GEOJSON), help="Directory containing nucleus GeoJSON")
    parser.add_argument(
        "--raw-output",
        default=str(MORPHOLOGY_FEATURES / "morphology_raw.parquet"),
    )
    parser.add_argument(
        "--output",
        default=str(MORPHOLOGY_FEATURES / "morphology_standardized.parquet"),
    )
    parser.add_argument(
        "--scaler-output",
        default=str(MORPHOLOGY_FEATURES / "scaler.joblib"),
    )
    parser.add_argument(
        "--manifest-output",
        default=str(MORPHOLOGY_FEATURES / "feature_manifest.json"),
    )
    parser.add_argument("--n_jobs", type=int, default=8)
    args = parser.parse_args()
    
    image_dir = Path(args.image_dir)
    geojson_dir = Path(args.geojson_dir)
    
    print(f"Image Dir: {image_dir}")
    print(f"GeoJSON Dir: {geojson_dir}")
    if not image_dir.exists() or not geojson_dir.exists():
        print(f"Error: {image_dir} or {geojson_dir} not found.")
        return
        
    geojson_files = list(geojson_dir.glob("*.geojson"))
    print(f"Found {len(geojson_files)} GeoJSON files.")
    
    tasks = []
    
    # Check images
    for gj_path in geojson_files:
        stem = gj_path.stem
        img_path = image_dir / f"{stem}.png"
        if img_path.exists():
            tasks.append((stem, img_path, gj_path))
            continue
        img_path = image_dir / f"{stem}.jpg" 
        if img_path.exists():
            tasks.append((stem, img_path, gj_path))
            
    print(f"Found {len(tasks)} valid image/geojson pairs.")
    
    if len(tasks) == 0:
        return

    print(f"Processing {len(tasks)} items in parallel ({args.n_jobs} jobs)...")
    results_list = Parallel(n_jobs=args.n_jobs)(
        delayed(process_wrapper)(task) for task in tqdm(tasks)
    )
    
    # Filter Nones
    valid_data = []
    valid_indices = []
    
    for stem, res in results_list:
        if res is not None:
            valid_data.append(res)
            valid_indices.append(stem)
            
    print(f"Successfully computed features for {len(valid_data)} tiles.")
    
    if not valid_data:
        print("No valid data produced.")
        return
        
    df = pd.DataFrame(valid_data, index=valid_indices)
    
    raw_output = Path(args.raw_output)
    standardized_output = Path(args.output)
    scaler_output = Path(args.scaler_output)
    manifest_output = Path(args.manifest_output)
    for path in (raw_output, standardized_output, scaler_output, manifest_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving raw features to {raw_output}...")
    df.to_parquet(raw_output)

    # Normalize. The caller must provide training-only tiles to avoid leakage.
    print("Normalizing (StandardScaler)...")
    scaler = StandardScaler()
    norm_vals = scaler.fit_transform(df)
    
    df_norm = pd.DataFrame(norm_vals, index=df.index, columns=df.columns)
    
    print(f"Saving standardized features to {standardized_output}...")
    df_norm.to_parquet(standardized_output)
    dump(scaler, scaler_output)
    with open(manifest_output, "w") as f:
        json.dump(
            {
                "feature_order": list(df.columns),
                "rows": len(df),
                "raw_table": str(raw_output),
                "standardized_table": str(standardized_output),
                "scaler": str(scaler_output),
                "warning": "Scaler is valid only if input tiles are the sealed training split.",
            },
            f,
            indent=2,
        )
    
    print("Done.")

if __name__ == "__main__":
    main()
