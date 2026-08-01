import glob
import json
from tqdm import tqdm

geojsons = glob.glob(r"data/interim/annotations/tcga_brca/geojson\*.geojson")

unique_classes = set()

for path in tqdm(geojsons[:500]):
    with open(path, 'r') as f:
        data = json.load(f)
        for feature in data:
            cls_name = feature.get("properties", {}).get("classification", {}).get("name")
            if cls_name:
                unique_classes.add(cls_name)

print("Unique Classes found:", unique_classes)
