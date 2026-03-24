import argparse
import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed

TYPE_MAP = {
    "Neoplastic": 0,
    "Inflammatory": 1,
    "Connective": 2,
    "Dead": 3,
    "Epithelial": 4
}
NUM_CLASSES = 5
IMG_SIZE = 512
SIGMA = 3.0

def process_single_geojson(geojson_path, out_dir):
    try:
        stem = os.path.splitext(os.path.basename(geojson_path))[0]
        out_file = os.path.join(out_dir, f"{stem}.npy")
        
        # Skip if exists
        if os.path.exists(out_file): return stem
        
        with open(geojson_path, 'r') as f:
            data = json.load(f)
            
        features = data if isinstance(data, list) else data.get('features', [])
        
        # Initialize map: (H, W, K)
        spatial_map = np.zeros((IMG_SIZE, IMG_SIZE, NUM_CLASSES), dtype=np.float32)
        
        for feature in features:
            props = feature.get('properties', {})
            cls_obj = props.get('classification', {})
            c_name = cls_obj.get('name', 'Unknown')
            
            c_idx = TYPE_MAP.get(c_name, -1)
            if c_idx == -1:
                # If unknown type is encountered, we skip or add a map? We ignore unknown.
                continue
                
            geom = feature.get('geometry', {})
            geom_type = geom.get('type')
            coords_list = geom.get('coordinates', [])
            
            polygons = []
            if geom_type == 'Polygon' and len(coords_list) > 0:
                polygons.append(np.array(coords_list[0], dtype=np.int32))
            elif geom_type == 'MultiPolygon':
                 for part in coords_list:
                     if len(part) > 0:
                         polygons.append(np.array(part[0], dtype=np.int32))
                         
            for poly in polygons:
                if len(poly) > 0:
                    # Calculate centroid
                    M = cv2.moments(poly) if 'cv2' in globals() else None
                    # Fallback to mean if cv2 is not imported or moments fail
                    mean_x = np.mean(poly[:, 0])
                    mean_y = np.mean(poly[:, 1])
                    cx, cy = int(round(mean_x)), int(round(mean_y))
                    
                    if 0 <= cx < IMG_SIZE and 0 <= cy < IMG_SIZE:
                        spatial_map[cy, cx, c_idx] += 1.0 # Accumulate
                        
        # Apply Gaussian filter per channel
        for i in range(NUM_CLASSES):
            if spatial_map[:, :, i].max() > 0:
                spatial_map[:, :, i] = gaussian_filter(spatial_map[:, :, i], sigma=SIGMA)
                # Normalize peak to 1.0
                c_max = spatial_map[:, :, i].max()
                if c_max > 0:
                    spatial_map[:, :, i] = spatial_map[:, :, i] / c_max
                    
        # Clip, scale to 0-255, and convert to uint8 to save massive space (512x512x5 float16 = 2.5MB, uint8 compressed = <50KB)
        spatial_map = (np.clip(spatial_map, 0, 1) * 255.0).astype(np.uint8)
        
        # Save compressed (often 90% sparse, so size drops drastically)
        out_file = os.path.join(out_dir, f"{stem}.npz")
        np.savez_compressed(out_file, map=spatial_map)
        return stem
        
    except Exception as e:
        print(f"Error processing {geojson_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="PathOGen/data", help="Path to PathOGen data dir")
    parser.add_argument("--n_jobs", type=int, default=8)
    args = parser.parse_args()
    
    geojson_dir = Path(args.data_dir) / "geojsons"
    out_dir = Path(args.data_dir) / "spatial_maps"
    
    if not geojson_dir.exists():
        print(f"Error: {geojson_dir} not found. Please ensure data is copied.")
        return
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    geojson_files = list(geojson_dir.glob("*.geojson"))
    print(f"Found {len(geojson_files)} GeoJSON files to process.")
    
    if len(geojson_files) == 0:
        return
        
    print(f"Generating spatial maps in parallel ({args.n_jobs} jobs)...")
    
    # Run in parallel
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_single_geojson)(str(f), str(out_dir)) for f in tqdm(geojson_files)
    )
    
    valid = [r for r in results if r is not None]
    print(f"Successfully generated {len(valid)} spatial maps.")

if __name__ == "__main__":
    main()
