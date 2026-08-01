import argparse
import os
import json
from pathlib import Path

from cpathogen.utils.paths import GENERATOR_MANIFESTS, TCGA_TILES

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles-dir", default=str(TCGA_TILES))
    parser.add_argument(
        "--output",
        default=str(GENERATOR_MANIFESTS / "metadata.jsonl"),
    )
    args = parser.parse_args()

    tiles_dir = Path(args.tiles_dir)
    metadata_file = Path(args.output)
    
    if not tiles_dir.exists():
        print(f"Error: {tiles_dir} does not exist.")
        return
        
    print(f"Scanning {tiles_dir} for images...")
    images = list(tiles_dir.glob("*.png")) + list(tiles_dir.glob("*.jpg"))
    
    if len(images) == 0:
        print("No images found.")
        return
        
    print(f"Found {len(images)} images. Writing metadata.jsonl...")
    
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, "w") as f:
        for img_path in images:
            # Hugging Face imagefolder resolves file_name relative to metadata.jsonl.
            rel_path = os.path.relpath(img_path, metadata_file.parent)
            
            entry = {
                "file_name": rel_path,
                "text": "he"  # Constant prompt for all tiles as per pathogen.txt
            }
            f.write(json.dumps(entry) + "\n")
            
    print(f"Successfully wrote {len(images)} entries to {metadata_file}.")

if __name__ == "__main__":
    main()
