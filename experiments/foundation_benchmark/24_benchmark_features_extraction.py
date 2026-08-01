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
from torchvision import transforms

# For CTransPath
import sys
sys.path.append(r".")
import ctran

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading UNI2-h model...")
    timm_kwargs = {
        'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
        'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667*2,
        'num_classes': 0, 'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 'dynamic_img_size': True
    }
    uni_model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    uni_transform = create_transform(**resolve_data_config(uni_model.pretrained_cfg, model=uni_model))
    uni_model.eval().to(device)

    print("Loading CTransPath model...")
    ctp_model = ctran.ctranspath()
    ctp_model.head = torch.nn.Identity() # Remove classification head if present
    
    # Load weights
    ckpt_path = r"artifacts/models/ctranspath.pth"
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if 'model' in state_dict:
        state_dict = state_dict['model']
        
    # Fix state_dict for newer timm (downsample layer indices shifted by 1)
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'downsample' in k:
            parts = k.split('.')
            if parts[0] == 'layers':
                layer_idx = int(parts[1])
                parts[1] = str(layer_idx + 1)
                new_k = '.'.join(parts)
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v
        else:
            new_state_dict[k] = v
            
    ctp_model.load_state_dict(new_state_dict, strict=False)
    ctp_model.eval().to(device)
    
    ctp_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    print("Loading molecular subtypes map...")
    CSV_PATH = r"data/raw/tcga_brca/molecular_subtypes.csv"
    df = pd.read_csv(CSV_PATH)
    subtype_map = {}
    for idx, row in df.iterrows():
        subtype = row['molecular_subtype']
        # Combine Luminal A and B
        if subtype in ['BRCA.Luminal A', 'BRCA.Luminal B']:
            subtype = 'BRCA.Luminal'
        subtype_map[row['sampleID']] = subtype

    print("Scanning images directory...")
    IMAGES_DIR = r"data/interim/tiles/tcga_brca"
    all_images = glob.glob(os.path.join(IMAGES_DIR, "*.png"))

    # Group all tiles by class (we want 500 per class total)
    class_to_tiles = {
        'BRCA.Basal-like': [],
        'BRCA.HER2-enriched': [],
        'BRCA.Luminal': [],
        'BRCA.Normal-like': []
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

    print("\nBalancing Dataset...")
    target_count = 500
    
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

    # --- FEATURE EXTRACTION ---
    uni_features = []
    ctp_features = []
    
    with torch.no_grad():
        for img_path in tqdm(selected_images, desc="Extracting Features"):
            img = Image.open(img_path).convert("RGB")
            
            # UNI-2h
            u_tensor = uni_transform(img).unsqueeze(0).to(device)
            with torch.amp.autocast('cuda'):
                u_out = uni_model(u_tensor)
            uni_features.append(u_out[0].cpu().numpy())
            
            # CTransPath
            c_tensor = ctp_transform(img).unsqueeze(0).to(device)
            with torch.amp.autocast('cuda'):
                c_out = ctp_model(c_tensor)
                if c_out.dim() == 4:
                    c_out = c_out.mean(dim=(1, 2))
                elif c_out.dim() == 3:
                    c_out = c_out.mean(dim=1)
            ctp_features.append(c_out[0].cpu().numpy())
            
    uni_features = np.array(uni_features)
    ctp_features = np.array(ctp_features)
    
    # Save to parquet
    print("\nSaving features to parquet...")
    df_out = pd.DataFrame({
        'image_path': selected_images,
        'label': selected_labels
    })
    
    # We can save embeddings as lists/arrays in parquet, but since it's 2D numpy arrays:
    df_out['uni2h'] = list(uni_features)
    df_out['ctranspath'] = list(ctp_features)
    
    df_out.to_parquet(r"data/processed/classification/tcga_subtypes/embeddings/combined/benchmark_features.parquet")
    print("Saved to benchmark_features.parquet")

if __name__ == "__main__":
    main()
