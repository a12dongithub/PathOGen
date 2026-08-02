import os
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np

def extract_features():
    DATA_DIR = Path(r"data/interim/tiles/bach")
    OUTPUT_FILE = Path(r"data/processed/classification/bach/embeddings/uni2/bach_uni2h_features.parquet")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load UNI2-h model as shown in virtual_staining.ipynb
    print("Loading UNI2-h model...")
    timm_kwargs = {
        'img_size': 224, 
        'patch_size': 14, 
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5, 
        'embed_dim': 1536,
        'mlp_ratio': 2.66667*2,
        'num_classes': 0, 
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 
        'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 
        'dynamic_img_size': True
    }
    uni_model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    uni_transform = create_transform(**resolve_data_config(uni_model.pretrained_cfg, model=uni_model))
    uni_model.eval()
    uni_model.to(device)
    
    classes = ["Normal", "Benign", "InSitu", "Invasive"]
    
    records = []
    
    for cls in classes:
        class_dir = DATA_DIR / cls
        if not class_dir.exists():
            continue
            
        img_files = list(class_dir.glob("*.png"))
        print(f"Extracting features for {cls}: {len(img_files)} tiles...")
        
        # We can process in batches if memory allows, but tile by tile is safer to avoid OOM
        # Let's do batch size of 16
        BATCH_SIZE = 16
        
        for i in tqdm(range(0, len(img_files), BATCH_SIZE), desc=cls):
            batch_files = img_files[i:i+BATCH_SIZE]
            
            batch_tensors = []
            valid_files = []
            
            for f in batch_files:
                try:
                    img = Image.open(f).convert('RGB')
                    tensor = uni_transform(img)
                    batch_tensors.append(tensor)
                    valid_files.append(f)
                except Exception as e:
                    print(f"Error reading {f}: {e}")
            
            if not batch_tensors:
                continue
                
            input_batch = torch.stack(batch_tensors).to(device)
            
            with torch.no_grad():
                # uni_model without classifier head outputs the embeddings
                # shape: (B, 1536)
                embeddings = uni_model(input_batch)
                
            embeddings_np = embeddings.cpu().numpy()
            
            for j, f in enumerate(valid_files):
                # Original filename is the stem without _rX_cY
                # e.g. b001_r0_c0 -> original is b001
                stem = f.stem
                orig_stem = stem.rsplit('_r', 1)[0]
                
                records.append({
                    "tile_filename": f.name,
                    "original_image": orig_stem,
                    "class": cls,
                    "embedding": embeddings_np[j].tolist()
                })
                
    df = pd.DataFrame(records)
    print(f"Saving {len(df)} records to {OUTPUT_FILE}...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE)
    print("Done!")

if __name__ == "__main__":
    extract_features()
