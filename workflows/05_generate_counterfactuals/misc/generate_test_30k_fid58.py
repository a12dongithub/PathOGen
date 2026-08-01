#!/usr/bin/env python
"""
Generate a few test images from the checkpoint-30000_FID58 model.
It uses SpatialCondEncoder for the spatial map and FiLM for the 16D morphology vector.
"""

import os
import random
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

import sys
sys.path.append(r"src")

# Import components from local training scripts
from cpathogen.generation.phase2 import SpatialCondEncoder, inject_film_into_unet
from cpathogen.generation.inference import generate_concat_conditioned
from cpathogen.evaluation.misc.legacy_fid.compare_checkpoints import (
    add_label,
    spatial_map_to_rgb_with_legend,
)

def main():
    CKPT_DIR = r"artifacts/runs/legacy_phase2_fid58/checkpoints/checkpoint-30000"
    BASE_MODEL = "Manojb/stable-diffusion-2-1-base"
    tiles_dir = Path("data/interim/tiles/tcga_brca")
    spatial_dir = Path("data/processed/conditions/spatial_maps")
    OUTPUT_DIR = Path(r"artifacts/runs/test_generation_fid58/samples")
    NUM_IMAGES = 4
    SEED = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading Base Models...")
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder", torch_dtype=weight_dtype).to(device)
    vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae", torch_dtype=weight_dtype).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")
    
    print(f"Loading UNet and applying FiLM...")
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet", torch_dtype=weight_dtype)
    
    # Expand conv_in to 8 channels (4 latent + 4 spatial)
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
    
    # Load Weights
    print("Loading Weights from checkpoint...")
    unet.load_state_dict(UNet2DConditionModel.from_pretrained(CKPT_DIR, subfolder="unet").state_dict(), strict=False)
    unet.to(device)
    
    spatial_encoder = SpatialCondEncoder().to(device, dtype=weight_dtype)
    spatial_encoder.load_state_dict(torch.load(os.path.join(CKPT_DIR, "spatial_encoder.pt"), map_location=device))
    
    film_mlps.load_state_dict(torch.load(os.path.join(CKPT_DIR, "film_mlps.pt"), map_location=device))
    film_mlps.to(device, dtype=weight_dtype)

    # ── Load Data ──
    print("Loading test data...")
    morph_path = Path("data/processed/conditions/morphology/standardized.parquet")
        
    morph_df = pd.read_parquet(morph_path)

    all_samples = []
    for file in tiles_dir.glob("*.png"):
        stem = file.stem
        spatial_path = spatial_dir / f"{stem}.npz"
        if spatial_path.exists() and stem in morph_df.index:
            all_samples.append((file, spatial_path, stem))

    random.seed(SEED)
    samples = random.sample(all_samples, min(NUM_IMAGES, len(all_samples)))

    real_images, spatial_maps_raw, morphologies, stems = [], [], [], []
    for img_path, spatial_path, stem in samples:
        real_images.append(Image.open(img_path).convert("RGB"))
        sm = np.load(spatial_path)["map"]
        spatial_maps_raw.append(sm)
        morphologies.append(torch.tensor(morph_df.loc[stem].values, dtype=torch.float32))
        stems.append(stem)

    print(f"Generating {len(samples)} images...")
    gen_images = generate_concat_conditioned(
        unet, vae, spatial_encoder, text_encoder, tokenizer,
        noise_scheduler, spatial_maps_raw, morphologies, device, weight_dtype,
        num_inference_steps=20, seed=SEED
    )

    print("Saving grids...")
    for idx in range(len(samples)):
        try:
            from compare_checkpoints import spatial_map_to_rgb_with_legend
            spatial_rgb = spatial_map_to_rgb_with_legend(spatial_maps_raw[idx] * 255.0)
        except:
            # Fallback if the function expects 0-255 already
            spatial_rgb = spatial_map_to_rgb_with_legend(spatial_maps_raw[idx])
            
        spatial_rgb = spatial_rgb.resize((512, 512), Image.NEAREST)
        real = real_images[idx].resize((512, 512))
        gen = gen_images[idx].resize((512, 512))

        spatial_labeled = add_label(spatial_rgb.copy(), "Spatial Map")
        real_labeled = add_label(real.copy(), "Real H&E")
        gen_labeled = add_label(gen.copy(), "Generated FID 58")

        grid = Image.new("RGB", (512 * 3, 512))
        grid.paste(spatial_labeled, (0, 0))
        grid.paste(real_labeled, (512, 0))
        grid.paste(gen_labeled, (1024, 0))
        
        grid_path = OUTPUT_DIR / f"{stems[idx]}_grid.png"
        grid.save(grid_path)
        print(f"Saved {grid_path}")

    print("Done!")

if __name__ == "__main__":
    main()
