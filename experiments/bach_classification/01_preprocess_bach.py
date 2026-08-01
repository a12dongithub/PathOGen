import os
import glob
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def preprocess_bach():
    DATA_DIR = Path(r"data/raw/bach/images")
    OUTPUT_DIR = Path(r"data/interim/tiles/bach")
    
    classes = ["Normal", "Benign", "InSitu", "Invasive"]
    
    TILE_SIZE = 512
    
    for cls in classes:
        class_dir = DATA_DIR / cls
        output_class_dir = OUTPUT_DIR / cls
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        tif_files = list(class_dir.glob("*.tif"))
        print(f"Processing {cls}: {len(tif_files)} images...")
        
        for tif_file in tqdm(tif_files, desc=cls):
            try:
                img = Image.open(tif_file)
                width, height = img.size
                
                # We expect 2048 x 1536 for BACH, which yields exactly 4x3 512x512 tiles.
                # If sizes vary slightly, we will just crop as many full 512x512 tiles as possible.
                n_cols = width // TILE_SIZE
                n_rows = height // TILE_SIZE
                
                for row in range(n_rows):
                    for col in range(n_cols):
                        left = col * TILE_SIZE
                        upper = row * TILE_SIZE
                        right = left + TILE_SIZE
                        lower = upper + TILE_SIZE
                        
                        tile = img.crop((left, upper, right, lower))
                        
                        tile_filename = f"{tif_file.stem}_r{row}_c{col}.png"
                        tile_path = output_class_dir / tile_filename
                        tile.save(tile_path)
            except Exception as e:
                print(f"Error processing {tif_file}: {e}")

if __name__ == "__main__":
    preprocess_bach()
    print("Preprocessing complete!")
