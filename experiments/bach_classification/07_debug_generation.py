import os
import random
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

import sys
sys.path.append(r"src")
from cpathogen.generation.phase2 import SpatialCondEncoder, inject_film_into_unet
from cpathogen.generation.inference import generate_concat_conditioned
from compare_checkpoints import spatial_map_to_rgb_with_legend, add_label

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16
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
    
    DATA_DIR = Path(r"data/processed/generator")
    OUTPUT_DIR = Path(r"artifacts/runs/debug_generation")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    morph_df = pd.read_parquet(Path("data/processed/generator/morphology_features/morphology_standardized.parquet"))
    
    samples = [
        "TCGA-A8-A083_x14336_y30720_BR",
        "TCGA-A2-A0CM_x16384_y51200_BR"
    ]
    
    for stem in samples:
        img_path = Path("data/interim/tiles/tcga_brca") / f"{stem}.png"
        spatial_path = Path("data/processed/generator/spatial_maps") / f"{stem}.npz"
        
        real_img = Image.open(img_path).convert("RGB")
        original_map = np.load(spatial_path)["map"]
        morphology = torch.tensor(morph_df.loc[stem].values, dtype=torch.float32)
        
        # PROPER INTERVENTION: Preserve the spatial locations of ALL cells, just change their class!
        # Take the max across all channels to get a single mask of "where cells are"
        cell_locations = np.max(original_map, axis=2)
        
        all_tumor_map = np.zeros_like(original_map)
        all_tumor_map[:, :, 0] = cell_locations
        
        all_immune_map = np.zeros_like(original_map)
        all_immune_map[:, :, 1] = cell_locations
        
        # Test generation with original
        gen_images_orig = generate_concat_conditioned(
            unet, vae, spatial_encoder, text_encoder, tokenizer, noise_scheduler,
            [original_map], [morphology], device, weight_dtype, num_inference_steps=20, seed=42
        )
        gen_img_orig = gen_images_orig[0]
        
        # Test generation with all tumor
        gen_images_tumor = generate_concat_conditioned(
            unet, vae, spatial_encoder, text_encoder, tokenizer, noise_scheduler,
            [all_tumor_map], [morphology], device, weight_dtype, num_inference_steps=20, seed=42
        )
        gen_img_tumor = gen_images_tumor[0]
        
        # Draw spatial maps
        try:
            spatial_rgb_orig = spatial_map_to_rgb_with_legend(original_map * 255.0).resize((512, 512), Image.NEAREST)
            spatial_rgb_tumor = spatial_map_to_rgb_with_legend(all_tumor_map * 255.0).resize((512, 512), Image.NEAREST)
        except:
            spatial_rgb_orig = spatial_map_to_rgb_with_legend(original_map).resize((512, 512), Image.NEAREST)
            spatial_rgb_tumor = spatial_map_to_rgb_with_legend(all_tumor_map).resize((512, 512), Image.NEAREST)
        
        # Assemble grid
        grid = Image.new("RGB", (512 * 5, 512))
        grid.paste(add_label(real_img.resize((512, 512)), "Real Image"), (0, 0))
        grid.paste(add_label(spatial_rgb_orig, "Orig Spatial Map"), (512, 0))
        grid.paste(add_label(gen_img_orig, "Gen Orig Map"), (1024, 0))
        grid.paste(add_label(spatial_rgb_tumor, "Tumor Spatial Map"), (1536, 0))
        grid.paste(add_label(gen_img_tumor, "Gen Tumor Map"), (2048, 0))
        
        grid_path = OUTPUT_DIR / f"{stem}_debug_grid_proper.png"
        grid.save(grid_path)
        print(f"Saved {grid_path}")

if __name__ == "__main__":
    main()
