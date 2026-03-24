import os
import json
from pathlib import Path

def main():
    data_dir = Path("./data")
    tiles_dir = data_dir / "tiles"
    metadata_file = data_dir / "metadata.jsonl"
    
    if not tiles_dir.exists():
        print(f"Error: {tiles_dir} does not exist.")
        return
        
    print(f"Scanning {tiles_dir} for images...")
    images = list(tiles_dir.glob("*.png")) + list(tiles_dir.glob("*.jpg"))
    
    if len(images) == 0:
        print("No images found.")
        return
        
    print(f"Found {len(images)} images. Writing metadata.jsonl...")
    
    with open(metadata_file, "w") as f:
        for img_path in images:
            # Hugging Face imagefolder expects path relative to metadata.jsonl if they are in subdirs
            # e.g., "file_name": "tiles/img1.png"
            rel_path = f"tiles/{img_path.name}"
            
            entry = {
                "file_name": rel_path,
                "text": "he"  # Constant prompt for all tiles as per pathogen.txt
            }
            f.write(json.dumps(entry) + "\n")
            
    print(f"Successfully wrote {len(images)} entries to {metadata_file}.")

if __name__ == "__main__":
    main()
