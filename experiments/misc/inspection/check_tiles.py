import os
import glob
import pandas as pd

CSV_PATH = r"data/raw/tcga_brca/molecular_subtypes.csv"
df = pd.read_csv(CSV_PATH)
subtype_map = {}
for idx, row in df.iterrows():
    subtype = row['molecular_subtype']
    if subtype in ['BRCA.Luminal A', 'BRCA.Luminal B']:
        subtype = 'BRCA.Luminal'
    subtype_map[row['sampleID']] = subtype

IMAGES_DIR = r"data/interim/tiles/tcga_brca"
all_images = glob.glob(os.path.join(IMAGES_DIR, "*.png"))

class_to_tiles = {
    'BRCA.Basal': [],
    'BRCA.Her2': [],
    'BRCA.Luminal': [],
    'BRCA.Normal': []
}

for img_path in all_images:
    basename = os.path.basename(img_path)
    if not basename.startswith("TCGA-"):
        continue
    parts = basename.split("_")[0].split("-")
    if len(parts) >= 3:
        patient_id = "-".join(parts[:3])
        if patient_id in subtype_map:
            st = subtype_map[patient_id]
            if st in class_to_tiles:
                class_to_tiles[st].append(img_path)
            else:
                print(f"Unknown subtype {st}")

for c, tiles in class_to_tiles.items():
    print(f"{c}: {len(tiles)} tiles")
