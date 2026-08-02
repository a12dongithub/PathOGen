#!/usr/bin/env python
"""
FID evaluation for Phase 2 ControlNet checkpoint (no FiLM).
Loads UNet, ControlNet, and VAE from an accelerator checkpoint directory,
generates images conditioned on spatial maps, and computes FID against real tiles.
"""

import os
import random
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from safetensors.torch import load_file as load_safetensors
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as T

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
    UNet2DConditionModel,
)

# Monkey-patch eigvals for CPU fallback (avoids CUDA linalg crash on some builds)
original_eigvals = torch.linalg.eigvals
def eigvals_patched(A):
    if A.is_cuda:
        return original_eigvals(A.cpu()).to(A.device)
    return original_eigvals(A)
torch.linalg.eigvals = eigvals_patched


# ─── Cell Type Colors for spatial map visualization ───
CELL_TYPES = [
    ("Neoplastic",           (255, 255, 255)),  # ch0 → White
    ("Inflammatory",         (0,   255, 255)),  # ch1 → Cyan
    ("Connective",           (0,   255, 0)),    # ch2 → Green
    ("Dead",                 (255, 255, 0)),    # ch3 → Yellow
    ("Non-Neoplastic Epi.",  (255, 128, 0)),    # ch4 → Orange
]


def spatial_map_to_rgb_with_legend(spatial_map_np):
    """Convert 5-ch spatial map to color-coded RGB with cell-type legend."""
    colors = np.array([c for _, c in CELL_TYPES], dtype=np.float32)
    h, w, c = spatial_map_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for ch in range(min(c, 5)):
        mask = spatial_map_np[:, :, ch] / 255.0
        rgb += mask[:, :, np.newaxis] * colors[ch][np.newaxis, np.newaxis, :]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb)

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    legend_items = []
    for ch in range(min(c, 5)):
        if spatial_map_np[:, :, ch].max() > 1:
            legend_items.append(CELL_TYPES[ch])

    if legend_items:
        y_start = h - 20 * len(legend_items) - 10
        box_w = 170
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([2, y_start - 4, box_w, h - 2], fill=(0, 0, 0, 180))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(img)

        for i, (name, color) in enumerate(legend_items):
            y = y_start + i * 20
            draw.ellipse([8, y + 3, 18, y + 13], fill=color)
            draw.text((24, y), name, fill="white", font=font)

    return img


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
    """Load the SD ControlNet pipeline from an accelerator checkpoint (no FiLM)."""
    # Recursively navigate into nested checkpoint dirs until we find unet/controlnet
    inner_path = ckpt_dir
    for _ in range(3):  # max 3 levels deep
        unet_candidate = os.path.join(inner_path, "unet")
        if os.path.exists(unet_candidate):
            break
        # Try checkpoint-* subdirs first, then any subdirs
        subdirs = [d for d in os.listdir(inner_path) if os.path.isdir(os.path.join(inner_path, d))]
        ckpt_dirs = [d for d in subdirs if d.startswith("checkpoint-")]
        if ckpt_dirs:
            # Use the latest checkpoint
            ckpt_dirs.sort(key=lambda x: int(x.split("-")[1]))
            inner_path = os.path.join(inner_path, ckpt_dirs[-1])
        elif subdirs:
            inner_path = os.path.join(inner_path, subdirs[0])
        else:
            break
    
    print(f"  Resolved checkpoint path: {inner_path}")
    unet_path = os.path.join(inner_path, "unet")
    controlnet_path = os.path.join(inner_path, "controlnet")
    vae_path = os.path.join(inner_path, "vae")

    print(f"  Loading UNet from: {unet_path}")
    unet = UNet2DConditionModel.from_pretrained(inner_path, subfolder="unet", torch_dtype=torch.float16)

    print(f"  Loading ControlNet from: {controlnet_path}")
    controlnet = ControlNetModel.from_pretrained(inner_path, subfolder="controlnet", torch_dtype=torch.float16)

    # Load trained VAE if available, otherwise use base model VAE
    if False and os.path.exists(vae_path):
        print(f"  Loading trained VAE from: {vae_path}")
        vae = AutoencoderKL.from_pretrained(inner_path, subfolder="vae", torch_dtype=torch.float16)
    else:
        print(f"  Using base model VAE (no trained VAE found)")
        vae = None

    print(f"  Building pipeline...")
    pipeline_kwargs = dict(
        unet=unet,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    if vae is not None:
        pipeline_kwargs["vae"] = vae

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_id, **pipeline_kwargs
    )
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def generate_all(pipeline, spatial_maps, device, seed, cond_scale, batch_size):
    """Generate images conditioned on spatial maps only (no morphology)."""
    generated = []
    for i in tqdm(range(0, len(spatial_maps), batch_size), desc="Generating"):
        end = min(i + batch_size, len(spatial_maps))

        spatial_tensor = torch.stack([
            torch.from_numpy(sm.astype(np.float32) / 255.0).permute(2, 0, 1)
            for sm in spatial_maps[i:end]
        ]).to(device, dtype=torch.float16)

        prompts = ["he"] * (end - i)
        generator = torch.Generator(device=device).manual_seed(seed + i)

        with torch.autocast("cuda"):
            outputs = pipeline(
                prompt=prompts,
                image=spatial_tensor,
                controlnet_conditioning_scale=cond_scale,
                num_inference_steps=20,
                generator=generator,
            ).images
        generated.extend(outputs)
    return generated


def main():
    # ── Configuration ──
    CKPT = "artifacts/runs/legacy_phase2_controlnet/checkpoints/checkpoint-10000"
    BASE_MODEL = "Manojb/stable-diffusion-2-1-base"
    tiles_dir = Path("data/interim/tiles/tcga_brca")
    spatial_dir = Path("data/processed/conditions/spatial_maps")
    OUTPUT_DIR = Path("artifacts/runs/legacy_phase2_controlnet/evaluation")
    NUM_IMAGES = 2000
    BATCH_SIZE = 8
    SEED = 42
    COND_SCALE = 0.5

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required. Exiting.")
        return
    device = torch.device("cuda")
    print(f"Using: {torch.cuda.get_device_name(0)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "grids").mkdir(exist_ok=True)
    (OUTPUT_DIR / "generated").mkdir(exist_ok=True)

    # ── Load Data ──

    # Collect all valid pairs (image + spatial map, no morphology needed)
    all_samples = []
    for file in tiles_dir.glob("*.png"):
        stem = file.stem
        spatial_path = spatial_dir / f"{stem}.npz"
        if spatial_path.exists():
            all_samples.append((file, spatial_path, stem))

    print(f"Found {len(all_samples)} total valid pairs.")

    # Random shuffle and take NUM_IMAGES
    random.seed(SEED + 777)
    random.shuffle(all_samples)
    samples = all_samples[:NUM_IMAGES]
    print(f"Randomly selected {len(samples)} samples.")

    # Pre-load all data
    print("Loading data...")
    real_images, spatial_maps_raw, stems = [], [], []
    for img_path, spatial_path, stem in tqdm(samples, desc="Loading"):
        real_images.append(Image.open(img_path).convert("RGB"))
        sm = np.load(spatial_path)["map"]  # uint8 (0-255)
        spatial_maps_raw.append(sm)
        stems.append(stem)

    # ── Load Pipeline ──
    print(f"\n{'='*60}")
    print(f"Loading Pipeline (no FiLM): {CKPT}")
    print(f"{'='*60}")
    pipeline = load_pipeline(BASE_MODEL, CKPT, device)

    # ── Generate ──
    print(f"\nGenerating {len(samples)} images...")
    gen_images = generate_all(pipeline, spatial_maps_raw, device, SEED, COND_SCALE, BATCH_SIZE)

    # Save individual generated images
    print("Saving generated images...")
    for idx, img in enumerate(gen_images):
        img.save(OUTPUT_DIR / "generated" / f"{idx:04d}_{stems[idx]}.png")

    # ── Create Comparison Grids (first 200 for visualization) ──
    num_grids = min(200, len(samples))
    print(f"\nCreating {num_grids} comparison grids...")
    for idx in tqdm(range(num_grids), desc="Grids"):
        spatial_rgb = spatial_map_to_rgb_with_legend(spatial_maps_raw[idx])
        spatial_rgb = spatial_rgb.resize((512, 512), Image.NEAREST)

        real = real_images[idx].resize((512, 512))
        gen = gen_images[idx].resize((512, 512))

        spatial_labeled = add_label(spatial_rgb.copy(), "Spatial Map")
        real_labeled = add_label(real.copy(), "Real H&E")
        gen_labeled = add_label(gen.copy(), "Phase2 ControlNet (no FiLM)")

        # 3-panel grid: [Spatial Map | Real H&E | Generated]
        grid = Image.new("RGB", (512 * 3, 512))
        grid.paste(spatial_labeled, (0, 0))
        grid.paste(real_labeled, (512, 0))
        grid.paste(gen_labeled, (1024, 0))
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
    print(f"  FID (Phase 2 ControlNet, no FiLM, {len(samples)} images): {score:.2f}")
    print(f"{'='*60}")
    print(f"\nDone!")
    print(f"  Grids:     {OUTPUT_DIR / 'grids'} ({num_grids} files)")
    print(f"  Generated: {OUTPUT_DIR / 'generated'} ({len(samples)} files)")

    # Save FID result
    with open(OUTPUT_DIR / "fid_result.txt", "w") as f:
        f.write(f"Checkpoint: {CKPT}\n")
        f.write(f"Images: {len(samples)}\n")
        f.write(f"Conditioning Scale: {COND_SCALE}\n")
        f.write(f"FID: {score:.4f}\n")


if __name__ == "__main__":
    main()
