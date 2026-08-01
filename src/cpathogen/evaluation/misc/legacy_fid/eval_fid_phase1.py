#!/usr/bin/env python
import os
import random
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as T

from diffusers import StableDiffusionPipeline, UNet2DConditionModel

def main():
    CKPT = "artifacts/runs/phase1_domain_adapt/checkpoints/checkpoint-30000/unet"
    BASE_MODEL = "Manojb/stable-diffusion-2-1-base"
    tiles_dir = Path("data/interim/tiles/tcga_brca")
    NUM_IMAGES = 2000
    BATCH_SIZE = 16
    SEED = 42

    device = torch.device("cuda")

    # Load Real Images
    all_files = list(tiles_dir.glob("*.png"))
    
    random.seed(SEED + 777)
    random.shuffle(all_files)
    samples = all_files[:NUM_IMAGES]
    
    print(f"Loading {len(samples)} real images...")
    real_images = [Image.open(f).convert("RGB") for f in tqdm(samples)]

    # Load Pipeline
    print(f"Loading Phase 1 UNet from {CKPT}...")
    unet = UNet2DConditionModel.from_pretrained(CKPT, torch_dtype=torch.float16)
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, unet=unet, torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)

    # Generate
    print(f"Generating {len(samples)} images unconditionally...")
    gen_images = []
    generator = torch.Generator(device=device).manual_seed(SEED)
    
    for i in tqdm(range(0, len(samples), BATCH_SIZE), desc="Generating"):
        end = min(i + BATCH_SIZE, len(samples))
        prompts = ["he"] * (end - i)
        with torch.autocast("cuda"):
            outputs = pipeline(
                prompt=prompts,
                num_inference_steps=20,
                generator=generator,
            ).images
        gen_images.extend(outputs)

    # Calculate FID
    print(f"Calculating FID...")
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    transform = T.Compose([T.Resize((299, 299)), T.ToTensor()])

    fid_batch = 64
    for i in tqdm(range(0, len(samples), fid_batch), desc="FID features"):
        end = min(i + fid_batch, len(samples))
        reals = torch.stack([transform(real_images[j]) for j in range(i, end)]).to(device)
        gens = torch.stack([transform(gen_images[j]) for j in range(i, end)]).to(device)
        fid.update(reals, real=True)
        fid.update(gens, real=False)

    score = fid.compute().item()
    print(f"\n============================================================")
    print(f"  FID (Phase 1 checkpoint-30000, {len(samples)} images): {score:.2f}")
    print(f"============================================================\n")

if __name__ == "__main__":
    main()
