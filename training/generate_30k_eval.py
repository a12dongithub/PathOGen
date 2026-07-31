#!/usr/bin/env python
"""
Generate 2000 images from checkpoint-30000 (Phase 1: pure SD UNet, no ControlNet/FiLM) + FID.
Publication-quality grids: [Real H&E | checkpoint-30000]
"""

import os
import random
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from safetensors.torch import load_file as load_safetensors
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as T

from diffusers import StableDiffusionPipeline, UNet2DConditionModel

def add_label(img, text):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (img.width - tw) // 2
    y = 4
    draw.rectangle([x - 4, y - 2, x + tw + 4, y + th + 2], fill="black")
    draw.text((x, y), text, fill="white", font=font)
    return img

def load_pipeline(base_model_id, ckpt_dir, device):
    inner_dirs = [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")]
    inner_path = os.path.join(ckpt_dir, inner_dirs[0]) if inner_dirs else ckpt_dir

    unet_path = os.path.join(inner_path, "unet")

    print(f"  Loading base UNet from: {base_model_id}")
    unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet", torch_dtype=torch.float16)

    print(f"  Loading trained weights from: {unet_path}")
    safetensors_path = os.path.join(unet_path, "diffusion_pytorch_model.safetensors")
    
    if os.path.exists(safetensors_path):
        ckpt_state = load_safetensors(safetensors_path)
    else:
        raise FileNotFoundError(f"No model weights found in {unet_path}")

    missing, unexpected = unet.load_state_dict(ckpt_state, strict=False)
    
    print(f"  Building pipeline...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        unet=unet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline

def generate_all(pipeline, num_images, device, seed, batch_size):
    generated = []
    for i in tqdm(range(0, num_images, batch_size), desc="Generating"):
        end = min(i + batch_size, num_images)
        prompts = ["he"] * (end - i)
        generator = torch.Generator(device=device).manual_seed(seed + i)

        with torch.autocast("cuda"):
            outputs = pipeline(
                prompt=prompts,
                num_inference_steps=20,
                generator=generator,
            ).images
        generated.extend(outputs)
    return generated

def main():
    CKPT = "./checkpoint-30000"
    BASE_MODEL = "Manojb/stable-diffusion-2-1-base"
    DATA_DIR = Path("../results/512_final_dataset")
    OUTPUT_DIR = Path("./checkpoint30k_results")
    NUM_IMAGES = 2000
    BATCH_SIZE = 8  # RTX 6000 48GB can handle 8
    SEED = 42

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required. Exiting.")
        return
    device = torch.device("cuda")
    print(f"Using: {torch.cuda.get_device_name(0)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "grids").mkdir(exist_ok=True)
    (OUTPUT_DIR / "generated").mkdir(exist_ok=True)

    # ── Load Data ──
    tiles_dir = DATA_DIR / "images"

    # Collect ALL valid valid files
    all_samples = []
    for file in tiles_dir.glob("*.png"):
        all_samples.append(file)

    print(f"Found {len(all_samples)} total valid images.")

    # Random shuffle and take NUM_IMAGES
    random.seed(SEED + 777)
    random.shuffle(all_samples)
    samples = all_samples[:NUM_IMAGES]
    print(f"Randomly selected {len(samples)} samples.")

    # Pre-load all data
    print("Loading data...")
    real_images, stems = [], []
    for img_path in tqdm(samples, desc="Loading"):
        real_images.append(Image.open(img_path).convert("RGB"))
        stems.append(img_path.stem)

    # ── Load Pipeline ──
    ckpt_name = os.path.basename(os.path.normpath(CKPT))
    print(f"\n{'='*60}")
    print(f"Loading Pipeline: {ckpt_name}")
    print(f"{'='*60}")
    pipeline = load_pipeline(BASE_MODEL, CKPT, device)

    # ── Generate ──
    print(f"\nGenerating {len(samples)} images from {ckpt_name}...")
    gen_images = generate_all(pipeline, len(samples), device, SEED, BATCH_SIZE)

    # Save individual generated images
    print("Saving generated images...")
    for idx, img in enumerate(gen_images):
        img.save(OUTPUT_DIR / "generated" / f"{idx:04d}_{stems[idx]}.png")

    # ── Create Comparison Grids (first 200 for visualization) ──
    num_grids = min(200, len(samples))
    print(f"\nCreating {num_grids} comparison grids...")
    for idx in tqdm(range(num_grids), desc="Grids"):
        real = real_images[idx].resize((512, 512))
        gen = gen_images[idx].resize((512, 512))

        real_labeled = add_label(real.copy(), "Real H&E")
        gen_labeled = add_label(gen.copy(), ckpt_name)

        # 2-panel grid: [Real H&E | Generated]
        grid = Image.new("RGB", (512 * 2, 512))
        grid.paste(real_labeled, (0, 0))
        grid.paste(gen_labeled, (512, 0))
        grid.save(OUTPUT_DIR / "grids" / f"{idx:04d}_{stems[idx]}.png")

    # ── Calculate FID ──
    print(f"\n{'='*60}")
    print(f"Calculating FID on {len(samples)} image pairs...")
    print(f"{'='*60}")

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    transform = T.Compose([T.Resize((299, 299)), T.ToTensor()])

    fid_batch = 32
    for i in tqdm(range(0, len(samples), fid_batch), desc="FID features"):
        end = min(i + fid_batch, len(samples))
        reals = torch.stack([transform(real_images[j]) for j in range(i, end)]).to(device)
        gens = torch.stack([transform(gen_images[j]) for j in range(i, end)]).to(device)
        fid.update(reals, real=True)
        fid.update(gens, real=False)

    score = fid.compute().item()

    print(f"\n{'='*60}")
    print(f"  FID (checkpoint-30000, {len(samples)} images): {score:.2f}")
    print(f"{'='*60}")
    print(f"\nDone!")
    print(f"  Grids:     {OUTPUT_DIR / 'grids'} ({num_grids} files)")
    print(f"  Generated: {OUTPUT_DIR / 'generated'} ({len(samples)} files)")

    # Save FID result
    with open(OUTPUT_DIR / "fid_result.txt", "w") as f:
        f.write(f"Checkpoint: {ckpt_name}\n")
        f.write(f"Images: {len(samples)}\n")
        f.write(f"FID: {score:.4f}\n")


if __name__ == "__main__":
    main()
