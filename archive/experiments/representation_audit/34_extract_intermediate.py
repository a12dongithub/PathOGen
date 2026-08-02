import os
import glob
import pandas as pd
import numpy as np
import torch
import random
from PIL import Image
from tqdm import tqdm
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    OUTPUT_DIR = r"data/misc/tcga_10k_cached_tensors"
    META_OUTPUT = r"data/processed/classification/tcga_subtypes/manifests/legacy_10k_samples.csv"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(META_OUTPUT), exist_ok=True)
    
    print("Loading UNI2-h model...")
    timm_kwargs = {
        'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
        'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667*2,
        'num_classes': 0, 'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 'dynamic_img_size': True
    }
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    model.eval().to(device)

    print("Loading molecular subtypes map...")
    CSV_PATH = r"data/raw/tcga_brca/molecular_subtypes.csv"
    df = pd.read_csv(CSV_PATH)
    subtype_map = {}
    for idx, row in df.iterrows():
        subtype = row['molecular_subtype']
        if subtype in ['BRCA.Luminal A', 'BRCA.Luminal B']:
            subtype = 'BRCA.Luminal'
        if pd.isna(subtype) or subtype == 'BRCA.Normal-like':
            continue
        subtype_map[row['sampleID']] = subtype

    print("Scanning images directory...")
    IMAGES_DIR = r"data/interim/tiles/tcga_brca"
    all_images = glob.glob(os.path.join(IMAGES_DIR, "*.png"))

    class_to_tiles = {
        'BRCA.Basal-like': [],
        'BRCA.HER2-enriched': [],
        'BRCA.Luminal': []
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

    print("\nBalancing Dataset for 10k Images...")
    target_count = 3334 # approx 10k total (3334 * 3 = 10002)
    
    selected_images = []
    selected_labels = []
    
    for c, tiles in class_to_tiles.items():
        if len(tiles) >= target_count:
            sampled_tiles = random.sample(tiles, target_count)
            print(f"  {c}: Pooled {len(tiles)} tiles -> Sampled {target_count} unique tiles.")
        else:
            sampled_tiles = random.choices(tiles, k=target_count)
            print(f"  {c}: Pooled {len(tiles)} tiles -> Oversampled to {target_count} tiles.")
            
        selected_images.extend(sampled_tiles)
        selected_labels.extend([c] * target_count)

    print(f"\nTotal selected images: {len(selected_images)}")

    # Layers to extract (0-indexed in timm depending on version, 
    # n=[1, 3, 6, 12, 23] gets outputs of 2nd, 4th, 7th, 13th, and 24th blocks)
    layers_to_extract = [1, 3, 6, 12, 23]
    
    # Save a metadata CSV
    meta_df = pd.DataFrame({'image_path': selected_images, 'label': selected_labels})
    meta_df.to_csv(META_OUTPUT, index=False)
    
    # Extract features
    with torch.no_grad():
        for i, img_path in enumerate(tqdm(selected_images, desc="Extracting Intermediate Layers")):
            img_name = os.path.basename(img_path).replace('.png', '')
            out_file = os.path.join(OUTPUT_DIR, f"{img_name}.pt")
            if os.path.exists(out_file):
                continue
                
            img = Image.open(img_path).convert("RGB")
            u_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.amp.autocast('cuda'):
                intermediates = model.get_intermediate_layers(u_tensor, n=layers_to_extract, return_prefix_tokens=True)
            
            # intermediate returns a tuple (patches, prefix) per requested layer
            saved_dict = {}
            for l_idx, (patches, prefix) in zip(layers_to_extract, intermediates):
                # patches: [1, 256, 1536], prefix: [1, 9, 1536]
                # Combine them into full sequence so we can forward-prop later
                # In timm, the full sequence is usually [prefix, patches]
                full_seq = torch.cat([prefix, patches], dim=1).squeeze(0).half().cpu() # shape [265, 1536]
                saved_dict[l_idx] = full_seq
            
            torch.save(saved_dict, out_file)

    print("Finished extracting all 10k intermediate features!")

if __name__ == "__main__":
    main()
