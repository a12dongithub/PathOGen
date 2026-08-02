import os
import shutil
import random
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def create_validation_split(num_samples=2000):
    """
    Randomly selects a fixed number of validation samples from the training set and 
    moves them to a dedicated validation directory for reproducible FID and visual evaluation.
    Properly handles .npz spatial maps and .parquet morphology dataframes.
    """
    base_dir = Path(".")
    data_dir = base_dir / "data"
    val_dir = base_dir / "data_val"
    
    # Source directories
    tiles_src = data_dir / "tiles"
    spatial_src = data_dir / "spatial_maps"
    parquet_src = data_dir / "morphology_features/morphology_stats.parquet"
    
    # Target directories
    val_tiles_dir = val_dir / "tiles"
    val_spatial_dir = val_dir / "spatial_maps"
    
    if val_tiles_dir.exists() and len(list(val_tiles_dir.glob("*.png"))) == num_samples:
        print(f"Validation directory already exists with {num_samples} samples. Skipping extraction.")
        return
    elif val_dir.exists():
        print(f"Clearing outdated validation directory at {val_dir}...")
        shutil.rmtree(val_dir)
        
    print(f"Creating highly deterministic validation set of {num_samples} samples...")
    
    for d in [val_tiles_dir, val_spatial_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # Get all available tiles
    all_tiles = list(tiles_src.glob("*.png"))
    if not all_tiles:
        print(f"Error: No tiles found in {tiles_src}")
        return
        
    if not parquet_src.exists():
        print(f"Error: Morphology parquet not found at {parquet_src}")
        return
        
    print(f"Loading morphology database from {parquet_src}...")
    df = pd.read_parquet(parquet_src)
    
    # We must ensure that the selected tiles actually have their spatial maps and morphology rows ready.
    valid_tiles = []
    print("Verifying available file triplets...")
    for t in tqdm(all_tiles, desc="Checking files"):
        stem = t.stem
        if (spatial_src / f"{stem}.npz").exists() and (stem in df.index):
            valid_tiles.append(t)
            
    if len(valid_tiles) < num_samples:
        print(f"Error: Only found {len(valid_tiles)} valid triplets! Need {num_samples}.")
        return
        
    # Select randomly (but deterministically via seed)
    random.seed(42)  # Critical for reproducible validation set
    selected_tiles = random.sample(valid_tiles, num_samples)
    selected_stems = [t.stem for t in selected_tiles]
    
    # Copy files
    print(f"Copying {num_samples} image/spatial pairs to {val_dir}...")
    for tile_path in tqdm(selected_tiles, desc="Copying validation data"):
        stem = tile_path.stem
        # Copy tile
        shutil.copy2(tile_path, val_tiles_dir / f"{stem}.png")
        # Copy spatial
        shutil.copy2(spatial_src / f"{stem}.npz", val_spatial_dir / f"{stem}.npz")
        
    # Save the filtered validation morphology parquet
    print("Extracting validation morphology dataframe...")
    val_df = df.loc[selected_stems]
    val_df.to_parquet(val_dir / "morphology_stats.parquet")
        
    # Create metadata.jsonl for validation
    with open(val_dir / "metadata.jsonl", "w") as f:
        for tile in selected_tiles:
            f.write(f'{{"file_name": "tiles/{tile.name}", "text": "he"}}\n')

    print(f"Successfully extracted {num_samples} validation samples!")

if __name__ == "__main__":
    create_validation_split()
