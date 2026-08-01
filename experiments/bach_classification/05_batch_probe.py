import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
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
    new_conv_in = torch.nn.Conv2d(
        8, old_conv_in.out_channels,
        kernel_size=old_conv_in.kernel_size,
        stride=old_conv_in.stride,
        padding=old_conv_in.padding,
    ).to(unet.device, dtype=weight_dtype)
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

def batch_probe():
    N_SAMPLES = 50
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
    
    DATA_DIR = Path(r"data/processed/generator")
    spatial_dir = Path("data/processed/generator/spatial_maps")
    all_spatial_files = list(spatial_dir.glob("*.npz"))
    
    morph_path = Path("data/processed/generator/morphology_features/morphology_standardized.parquet")
    morph_df = pd.read_parquet(morph_path)
    
    valid_files = [f for f in all_spatial_files if f.stem in morph_df.index]
    print(f"Found {len(valid_files)} valid tiles with morphology.")
    
    random.seed(42)
    sample_files = random.sample(valid_files, min(N_SAMPLES, len(valid_files)))
    
    OUTPUT_DIR = Path(r"artifacts/runs/batch_probe/results")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    print(f"Starting batch probing on {len(sample_files)} tiles...")
    
    for idx, spatial_path in enumerate(tqdm(sample_files, desc="Batch Probing")):
        stem = spatial_path.stem
        try:
            original_map = np.load(spatial_path)["map"] # shape: (5, 512, 512)
            morphology = torch.tensor(morph_df.loc[stem].values, dtype=torch.float32)
            
            # Preserve spatial sparsity by finding all existing cells
            cell_locations = np.max(original_map, axis=2)
            
            all_tumor_map = np.zeros_like(original_map)
            all_tumor_map[:, :, 0] = cell_locations
            
            all_immune_map = np.zeros_like(original_map)
            all_immune_map[:, :, 1] = cell_locations
            
            all_stroma_map = np.zeros_like(original_map)
            all_stroma_map[:, :, 2] = cell_locations
            
            experiments = [
                ("Original", original_map),
                ("All Tumor", all_tumor_map),
                ("All Immune", all_immune_map),
                ("All Stroma", all_stroma_map)
            ]
            
            # Run sequentially to avoid OOM
            for name, sp_map in experiments:
                # Generate 1 image
                gen_images = generate_concat_conditioned(
                    unet, vae, spatial_encoder, text_encoder, tokenizer,
                    noise_scheduler, [sp_map], [morphology], device, weight_dtype,
                    num_inference_steps=20, seed=42
                )
                gen_img = gen_images[0]
                
                # Save a few random images to check quality later
                if idx < 5:
                    gen_img.save(OUTPUT_DIR / f"{stem}_{name.replace(' ', '_')}.png")
                
                # Extract features and predict
                tensor = uni_transform(gen_img.convert('RGB')).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = uni_model(tensor).cpu().numpy()
                    
                embedding_scaled = scaler.transform(embedding)
                probs = clf.predict_proba(embedding_scaled)[0]
                pred_idx = clf.predict(embedding_scaled)[0]
                pred_class = classes[pred_idx]
                
                result_row = {
                    "Sample_ID": stem,
                    "Intervention": name,
                    "Predicted_Class": pred_class,
                }
                for c, p in zip(classes, probs):
                    result_row[f"Prob_{c}"] = p
                    
                results.append(result_row)
                
        except Exception as e:
            print(f"Error processing {stem}: {e}")
            
    df_results = pd.DataFrame(results)
    output_csv = OUTPUT_DIR / "batch_probe_results.csv"
    df_results.to_csv(output_csv, index=False)
    print(f"\nSaved aggregated results to {output_csv}")

if __name__ == "__main__":
    batch_probe()
