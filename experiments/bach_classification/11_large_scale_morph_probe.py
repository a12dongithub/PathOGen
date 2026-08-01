import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import joblib
import random
from tqdm import tqdm

import sys
sys.path.append(r"src")
from cpathogen.generation.phase2 import SpatialCondEncoder, inject_film_into_unet
from cpathogen.generation.inference import generate_concat_conditioned

def load_pathogen_models(device, weight_dtype):
    CKPT_DIR = r"artifacts/runs/legacy_phase2_fid58/checkpoints/checkpoint-30000"
    BASE_MODEL = "Manojb/stable-diffusion-2-1-base"
    
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder", torch_dtype=weight_dtype).to(device)
    vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae", torch_dtype=weight_dtype).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")
    
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet", torch_dtype=weight_dtype)
    old_conv_in = unet.conv_in
    new_conv_in = torch.nn.Conv2d(8, old_conv_in.out_channels, kernel_size=old_conv_in.kernel_size,
                                  stride=old_conv_in.stride, padding=old_conv_in.padding).to(unet.device, dtype=weight_dtype)
    with torch.no_grad():
        new_conv_in.weight[:, :4] = old_conv_in.weight
        new_conv_in.weight[:, 4:] = 0.0
        new_conv_in.bias.copy_(old_conv_in.bias)
    unet.conv_in = new_conv_in
    unet.config['in_channels'] = 8

    film_mlps = inject_film_into_unet(unet, film_dim=16)
    unet.load_state_dict(UNet2DConditionModel.from_pretrained(CKPT_DIR, subfolder="unet").state_dict(), strict=False)
    unet.to(device)
    
    spatial_encoder = SpatialCondEncoder().to(device, dtype=weight_dtype)
    spatial_encoder.load_state_dict(torch.load(os.path.join(CKPT_DIR, "spatial_encoder.pt"), map_location=device))
    film_mlps.load_state_dict(torch.load(os.path.join(CKPT_DIR, "film_mlps.pt"), map_location=device))
    film_mlps.to(device, dtype=weight_dtype)
    
    return unet, vae, spatial_encoder, text_encoder, tokenizer, noise_scheduler

def load_uni2h_model(device):
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
    return uni_model, uni_transform

def add_prediction_label(img, class_name, probability, swap_name):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except:
        font = ImageFont.load_default()
    
    # Draw a black rectangle at the top for visibility
    draw.rectangle([0, 0, img.width, 50], fill=(0, 0, 0, 200))
    draw.text((10, 5), f"{class_name}: {probability*100:.1f}%", fill="white", font=font)
    
    # Optional: draw a smaller text for the swap name below it
    try:
        font_small = ImageFont.truetype("arial.ttf", 12)
    except:
        font_small = ImageFont.load_default()
    draw.text((10, 32), f"Morph: {swap_name[:20]}...", fill="lightgray", font=font_small)
    
    return img

def large_scale_morph_probe():
    N_TILES = 100
    N_SWAPS = 20
    BATCH_SIZE = 5 # GPU batch size for generator
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_dtype = torch.float16
    
    print("Loading PathOGen generative models...")
    unet, vae, spatial_encoder, text_encoder, tokenizer, noise_scheduler = load_pathogen_models(device, weight_dtype)
    
    print("Loading UNI2-h model...")
    uni_model, uni_transform = load_uni2h_model(device)
    
    print("Loading Classifier and Scaler...")
    MODEL_FILE = Path(r"artifacts/models/downstream/classifier.joblib")
    SCALER_FILE = Path(r"artifacts/models/downstream/scaler.joblib")
    clf = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    classes = ["Normal", "Benign", "InSitu", "Invasive"]
    
    DATA_DIR = Path(r"data/processed/conditions")
    spatial_dir = Path("data/processed/conditions/spatial_maps")
    all_spatial_files = list(spatial_dir.glob("*.npz"))
    
    morph_path = Path("data/processed/conditions/morphology/standardized.parquet")
    morph_df = pd.read_parquet(morph_path)
    
    valid_files = [f for f in all_spatial_files if f.stem in morph_df.index]
    
    # Set seed 456 for a different random 100 tiles
    random.seed(456)
    sample_files = random.sample(valid_files, min(N_TILES, len(valid_files)))
    all_stems = list(morph_df.index)
    
    OUTPUT_DIR = Path(r"artifacts/runs/large_scale_morph_probe")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grids_dir = OUTPUT_DIR / "grids"
    grids_dir.mkdir(exist_ok=True)
    
    results = []
    
    print(f"Starting Massive Morphological Probing on {len(sample_files)} tiles...")
    
    for idx, spatial_path in enumerate(tqdm(sample_files, desc="Tiles")):
        stem = spatial_path.stem
        try:
            # We keep the spatial map exact same for all 20 swaps
            original_map = np.load(spatial_path)["map"]
            
            # Select 20 random morphological embeddings
            swap_stems = random.sample(all_stems, N_SWAPS)
            
            swap_morphs = []
            for s in swap_stems:
                swap_morphs.append(torch.tensor(morph_df.loc[s].values, dtype=torch.float32))
                
            grid_images = []
            
            # Process generation in batches to prevent OOM
            for i in range(0, N_SWAPS, BATCH_SIZE):
                batch_morphs = swap_morphs[i:i+BATCH_SIZE]
                batch_maps = [original_map] * len(batch_morphs)
                
                gen_images = generate_concat_conditioned(
                    unet, vae, spatial_encoder, text_encoder, tokenizer,
                    noise_scheduler, batch_maps, batch_morphs, device, weight_dtype,
                    num_inference_steps=20, seed=42
                )
                
                # Predict and annotate each image in batch
                for j, gen_img in enumerate(gen_images):
                    swap_idx = i + j
                    swap_stem = swap_stems[swap_idx]
                    
                    # Predict
                    tensor = uni_transform(gen_img.convert('RGB')).unsqueeze(0).to(device)
                    with torch.no_grad():
                        embedding = uni_model(tensor).cpu().numpy()
                        
                    embedding_scaled = scaler.transform(embedding)
                    probs = clf.predict_proba(embedding_scaled)[0]
                    pred_idx = clf.predict(embedding_scaled)[0]
                    pred_class = classes[pred_idx]
                    prob_val = probs[pred_idx]
                    
                    # Record
                    result_row = {
                        "Target_Spatial_Stem": stem,
                        "Swap_Morphology_Stem": swap_stem,
                        "Predicted_Class": pred_class,
                        "Predicted_Prob": prob_val
                    }
                    for c, p in zip(classes, probs):
                        result_row[f"Prob_{c}"] = p
                    results.append(result_row)
                    
                    # Annotate image
                    labeled_img = add_prediction_label(gen_img.copy(), pred_class, prob_val, swap_stem)
                    grid_images.append(labeled_img)
            
            # Assemble 5 columns x 4 rows grid (20 images)
            grid = Image.new("RGB", (512 * 5, 512 * 4))
            for i, img in enumerate(grid_images):
                row = i // 5
                col = i % 5
                grid.paste(img, (512 * col, 512 * row))
                
            grid.save(grids_dir / f"{stem}_morph_swaps_grid.png")
            
        except Exception as e:
            print(f"Error processing {stem}: {e}")
            
    df_results = pd.DataFrame(results)
    output_csv = OUTPUT_DIR / "large_scale_morph_results.csv"
    df_results.to_csv(output_csv, index=False)
    print(f"\nSaved aggregated results to {output_csv}")

if __name__ == "__main__":
    large_scale_morph_probe()
