#!/usr/bin/env python
"""
Generate 2000 images from checkpoint-15000 with corrected spatial map labels + FID.
Publication-quality grids: [Spatial Map (labeled) | Real H&E | checkpoint-15000]

CellViT/PanNuke channel order (verified from data):
  ch0 = Neoplastic       → White
  ch1 = Inflammatory     → Cyan
  ch2 = Connective       → Green
  ch3 = Dead             → Yellow
  ch4 = Non-Neoplastic   → Orange
  Background (all 0)     → Black
"""

import os
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from safetensors.torch import load_file as load_safetensors
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as T

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UNet2DConditionModel,
)

# Monkey-patch eigvals for CPU fallback
original_eigvals = torch.linalg.eigvals
def eigvals_patched(A):
    if A.is_cuda:
        return original_eigvals(A.cpu()).to(A.device)
    return original_eigvals(A)
torch.linalg.eigvals = eigvals_patched


# ─── FiLM MLP ───
class FiLM_MLP(nn.Module):
    def __init__(self, in_dim=16, out_dim=320):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim * 2),
        )

    def forward(self, x):
        gamma_beta = self.net(x)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.clamp(-0.5, 0.5)
        beta = beta.clamp(-0.5, 0.5)
        return gamma.unsqueeze(-1).unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)


def inject_film_into_unet(unet, film_dim=16):
    film_mlps = nn.ModuleList()
    for name, module in unet.named_modules():
        if module.__class__.__name__ == "ResnetBlock2D":
            channels = module.out_channels
            mlp = FiLM_MLP(film_dim, channels)
            film_mlps.append(mlp)
            module.original_forward = module.forward
            module.film_mlp = mlp

            def new_forward(self, hidden_states, temb=None, **kwargs):
                out = self.original_forward(hidden_states, temb, **kwargs)
                if hasattr(self, "current_morph16") and self.current_morph16 is not None:
                    gamma, beta = self.film_mlp(self.current_morph16)
                    out = (1.0 + gamma) * out + beta
                return out

            module.forward = new_forward.__get__(module, module.__class__)
    return film_mlps


# ─── Corrected Cell Type Colors (verified from spatial map channel inspection) ───
CELL_TYPES = [
    ("Neoplastic",           (255, 255, 255)),  # ch0 → White
    ("Inflammatory",         (0,   255, 255)),  # ch1 → Cyan
    ("Connective",           (0,   255, 0)),    # ch2 → Green
    ("Dead",                 (255, 255, 0)),    # ch3 → Yellow
    ("Non-Neoplastic Epi.",  (255, 128, 0)),    # ch4 → Orange
]


def spatial_map_to_rgb_with_legend(spatial_map_np):
    """Convert 5-ch spatial map to color-coded RGB with corrected cell-type legend."""
    colors = np.array([c for _, c in CELL_TYPES], dtype=np.float32)
    h, w, c = spatial_map_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for ch in range(min(c, 5)):
        mask = spatial_map_np[:, :, ch] / 255.0  # normalize to 0-1
        rgb += mask[:, :, np.newaxis] * colors[ch][np.newaxis, np.newaxis, :]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb)

    # Draw legend
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    # Only show channels that are present
    legend_items = []
    for ch in range(min(c, 5)):
        if spatial_map_np[:, :, ch].max() > 1:
            legend_items.append(CELL_TYPES[ch])

    if legend_items:
        y_start = h - 20 * len(legend_items) - 10
        box_w = 170
        # Semi-transparent background
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
    inner_dirs = [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")]
    inner_path = os.path.join(ckpt_dir, inner_dirs[0]) if inner_dirs else ckpt_dir

    unet_path = os.path.join(inner_path, "unet")
    controlnet_path = os.path.join(inner_path, "controlnet")

    print(f"  Loading base UNet from: {base_model_id}")
    unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet", torch_dtype=torch.float16)

    print("  Injecting FiLM MLPs...")
    film_mlps = inject_film_into_unet(unet, film_dim=16)

    print(f"  Loading trained weights from: {unet_path}")
    safetensors_path = os.path.join(unet_path, "diffusion_pytorch_model.safetensors")
    bin_path = os.path.join(unet_path, "model.safetensors")
    pytorch_bin = os.path.join(unet_path, "diffusion_pytorch_model.bin")

    if os.path.exists(safetensors_path):
        ckpt_state = load_safetensors(safetensors_path)
    elif os.path.exists(bin_path):
        ckpt_state = load_safetensors(bin_path)
    elif os.path.exists(pytorch_bin):
        ckpt_state = torch.load(pytorch_bin, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model weights found in {unet_path}")

    missing, unexpected = unet.load_state_dict(ckpt_state, strict=False)
    film_loaded = sum(1 for k in ckpt_state if "film_mlp" in k)
    print(f"  Loaded {film_loaded} FiLM weight tensors")

    print(f"  Loading ControlNet from: {controlnet_path}")
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

    print(f"  Building pipeline...")
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_id,
        unet=unet,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline, film_mlps


def generate_all(pipeline, spatial_maps, morphologies, device, seed, cond_scale, batch_size):
    generated = []
    for i in tqdm(range(0, len(spatial_maps), batch_size), desc="Generating"):
        end = min(i + batch_size, len(spatial_maps))

        spatial_tensor = torch.stack([
            torch.from_numpy(sm.astype(np.float32) / 255.0).permute(2, 0, 1)
            for sm in spatial_maps[i:end]
        ]).to(device, dtype=torch.float16)

        morph_tensor = torch.stack(morphologies[i:end]).to(device, dtype=torch.float16)
        morph_dup = torch.cat([morph_tensor, morph_tensor], dim=0)
        for module in pipeline.unet.modules():
            if hasattr(module, "current_morph16"):
                module.current_morph16 = morph_dup

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
    CKPT = "./checkpoint-15000"
    BASE_MODEL = "Manojb/stable-diffusion-2-1-base"
    DATA_DIR = Path("../results/512_final_dataset")
    OUTPUT_DIR = Path("./checkpoint15k_results")
    NUM_IMAGES = 2000
    BATCH_SIZE = 8  # RTX 6000 48GB can handle 8
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
    tiles_dir = DATA_DIR / "images"
    spatial_dir = DATA_DIR / "spatial_maps"
    morph_path = DATA_DIR / "morphology_stats.parquet"
    if not morph_path.exists():
        morph_path = DATA_DIR / "morphology_features" / "morphology_stats.parquet"

    morph_df = pd.read_parquet(morph_path)

    # Collect ALL valid triplets
    all_samples = []
    for file in tiles_dir.glob("*.png"):
        stem = file.stem
        spatial_path = spatial_dir / f"{stem}.npz"
        if spatial_path.exists() and stem in morph_df.index:
            all_samples.append((file, spatial_path, stem))

    print(f"Found {len(all_samples)} total valid triplets.")

    # Random shuffle and take NUM_IMAGES
    random.seed(SEED + 777)
    random.shuffle(all_samples)
    samples = all_samples[:NUM_IMAGES]
    print(f"Randomly selected {len(samples)} samples.")

    # Pre-load all data
    print("Loading data...")
    real_images, spatial_maps_raw, morphologies, stems = [], [], [], []
    for img_path, spatial_path, stem in tqdm(samples, desc="Loading"):
        real_images.append(Image.open(img_path).convert("RGB"))
        sm = np.load(spatial_path)["map"]  # Keep as uint8 (0-255)
        spatial_maps_raw.append(sm)
        morphologies.append(torch.tensor(morph_df.loc[stem].values, dtype=torch.float32))
        stems.append(stem)

    # ── Load Pipeline ──
    ckpt_name = os.path.basename(os.path.normpath(CKPT))
    print(f"\n{'='*60}")
    print(f"Loading Pipeline: {ckpt_name}")
    print(f"{'='*60}")
    pipeline, film_mlps = load_pipeline(BASE_MODEL, CKPT, device)

    # ── Generate ──
    print(f"\nGenerating {len(samples)} images from {ckpt_name}...")
    gen_images = generate_all(pipeline, spatial_maps_raw, morphologies, device, SEED, COND_SCALE, BATCH_SIZE)

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
        gen_labeled = add_label(gen.copy(), ckpt_name)

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
    print(f"  FID (checkpoint-15000, {len(samples)} images): {score:.2f}")
    print(f"{'='*60}")
    print(f"\nDone!")
    print(f"  Grids:     {OUTPUT_DIR / 'grids'} ({num_grids} files)")
    print(f"  Generated: {OUTPUT_DIR / 'generated'} ({len(samples)} files)")

    # Save FID result
    with open(OUTPUT_DIR / "fid_result.txt", "w") as f:
        f.write(f"Checkpoint: {ckpt_name}\n")
        f.write(f"Images: {len(samples)}\n")
        f.write(f"Conditioning Scale: {COND_SCALE}\n")
        f.write(f"FID: {score:.4f}\n")


if __name__ == "__main__":
    main()
