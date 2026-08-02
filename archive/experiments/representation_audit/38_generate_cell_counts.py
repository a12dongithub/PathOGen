import os
import glob
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

def main():
    DATA_DIR = r"data/misc/tcga_10k_cached_tensors"
    META_PATH = r"data/processed/classification/tcga_subtypes/manifests/legacy_10k_samples.csv"
    MODEL_DIR = r"artifacts/runs/legacy_representation_audit/models"
    METRICS_DIR = r"artifacts/runs/legacy_representation_audit/metrics"
    FIGURES_DIR = r"artifacts/runs/legacy_representation_audit/figures"
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    GEOJSON_DIR = r"data/interim/annotations/tcga_brca/geojson"
    MORPH_PATH = r"data/processed/conditions/morphology/standardized.parquet"
    OUT_MORPH_PATH = r"data/processed/classification/tcga_subtypes/features/morphology_with_counts.parquet"
    
    # Load existing morph stats
    morph_df = pd.read_parquet(MORPH_PATH)
    
    # Load metadata to get the 10k list
    meta_df = pd.read_csv(META_PATH)
    
    # Classes we care about
    classes = ['Neoplastic', 'Inflammatory', 'Connective', 'Epithelial']
    
    # Create empty columns
    count_cols = ['count_tumor', 'count_immune', 'count_stroma', 'count_epithelial']
    for col in count_cols:
        if col not in morph_df.columns:
            morph_df[col] = 0.0
            
    print(f"Total tiles in morph_df: {len(morph_df)}")
    
    # Iterate over all geojsons that match the 10k dataset
    not_found = 0
    for idx, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Parsing GeoJSONs"):
        img_name = os.path.basename(row['image_path']).replace('.png', '')
        geojson_path = os.path.join(GEOJSON_DIR, f"{img_name}.geojson")
        
        if not os.path.exists(geojson_path):
            not_found += 1
            continue
            
        with open(geojson_path, 'r') as f:
            data = json.load(f)
            
        counts = {'Neoplastic': 0, 'Inflammatory': 0, 'Connective': 0, 'Epithelial': 0}
        
        for feature in data:
            cls_name = feature.get("properties", {}).get("classification", {}).get("name")
            if cls_name in counts:
                counts[cls_name] += 1
                
        # Update the dataframe
        if img_name in morph_df.index:
            morph_df.at[img_name, 'count_tumor'] = float(counts['Neoplastic'])
            morph_df.at[img_name, 'count_immune'] = float(counts['Inflammatory'])
            morph_df.at[img_name, 'count_stroma'] = float(counts['Connective'])
            morph_df.at[img_name, 'count_epithelial'] = float(counts['Epithelial'])
            
    print(f"Missing GeoJSONs for {not_found} tiles out of {len(meta_df)}")
    
    # Save the augmented morphology dataframe
    os.makedirs(os.path.dirname(OUT_MORPH_PATH), exist_ok=True)
    morph_df.to_parquet(OUT_MORPH_PATH)
    print(f"Saved augmented morph stats to {OUT_MORPH_PATH}")
    print(f"New morph_df shape: {morph_df.shape}")
    
if __name__ == "__main__":
    main()
