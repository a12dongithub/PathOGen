#!/usr/bin/env python
"""
Compare Phase 2 Checkpoints — Random Sample Comparison with Labeled Spatial Maps (GPU only)

Generates 100 RANDOM images from two checkpoints using identical noise but varying
spatial maps and morphology vectors. Spatial maps have a color-coded legend.

Grid: [Spatial Map (with legend) | Real Image | Checkpoint A | Checkpoint B]
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

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UNet2DConditionModel,
)


# ─── FiLM MLP (must match train_pathogen.py) ───
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


# Cell type colors and names
CELL_TYPES = [
    ("Background",    (40,  40,  40)),
    ("Neoplastic",    (255, 0,   0)),
    ("Inflammatory",  (0,   128, 255)),
    ("Connective",    (0,   200, 0)),
    ("Necrosis",      (255, 255, 0)),
]


def spatial_map_to_rgb_with_legend(spatial_map_np):
    """Convert 5-ch spatial map to color-coded RGB with a cell-type legend."""
    colors = np.array([c for _, c in CELL_TYPES], dtype=np.float32)
    h, w, c = spatial_map_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for ch in range(min(c, 5)):
        mask = spatial_map_np[:, :, ch]
        rgb += mask[:, :, np.newaxis] * colors[ch][np.newaxis, np.newaxis, :]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb)

    # Draw legend
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except:
        font = ImageFont.load_default()

    # Check which channels are present (non-zero)
    legend_items = []
    for ch in range(min(c, 5)):
        if spatial_map_np[:, :, ch].max() > 0.01:
            legend_items.append(CELL_TYPES[ch])

    # Draw legend box at bottom-left
    y_start = h - 18 * len(legend_items) - 8
    box_w = 150
    draw.rectangle([2, y_start - 2, box_w, h - 2], fill=(0, 0, 0, 180))

    for i, (name, color) in enumerate(legend_items):
        y = y_start + i * 18
        # Color swatch
        draw.rectangle([6, y + 2, 18, y + 14], fill=color)
        # Label
        draw.text((22, y), name, fill="white", font=font)

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

    print(f"  Loading trained UNet+FiLM weights from: {unet_path}")
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
    for i in tqdm(range(0, len(spatial_maps), batch_size)):
        end = min(i + batch_size, len(spatial_maps))

        spatial_tensor = torch.stack([
            torch.from_numpy(sm).float().permute(2, 0, 1)
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
    CKPT_A = "./checkpoint-10000"
    CKPT_B = "./checkpoint-15000"
    BASE_MODEL = "Manojb/stable-diffusion-2-1-base"
    DATA_DIR = Path("../results/512_final_dataset")
    OUTPUT_DIR = Path("./comparison_output_random")
    NUM_IMAGES = 100
    BATCH_SIZE = 4
    SEED = 42
    COND_SCALE = 0.5

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU is required. Exiting.")
        return
    device = torch.device("cuda")
    print(f"Using device: {torch.cuda.get_device_name(0)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "grids").mkdir(exist_ok=True)
    (OUTPUT_DIR / "ckpt_a").mkdir(exist_ok=True)
    (OUTPUT_DIR / "ckpt_b").mkdir(exist_ok=True)

    # ── Load Data ──
    tiles_dir = DATA_DIR / "images"
    if not tiles_dir.exists():
        tiles_dir = DATA_DIR / "tiles"
    spatial_dir = DATA_DIR / "spatial_maps"
    morph_path = DATA_DIR / "morphology_stats.parquet"
    if not morph_path.exists():
        morph_path = DATA_DIR / "morphology_features" / "morphology_stats.parquet"

    morph_df = pd.read_parquet(morph_path)

    # Collect ALL valid triplets first, then RANDOMLY sample
    all_samples = []
    for file in tiles_dir.glob("*.png"):
        stem = file.stem
        spatial_path = spatial_dir / f"{stem}.npz"
        if spatial_path.exists() and stem in morph_df.index:
            all_samples.append((file, spatial_path, stem))

    print(f"Found {len(all_samples)} total valid triplets.")

    # RANDOM SHUFFLE then take NUM_IMAGES
    random.seed(SEED + 999)  # Different seed from generation so it's truly random
    random.shuffle(all_samples)
    samples = all_samples[:NUM_IMAGES]
    print(f"Randomly selected {len(samples)} samples for comparison.")

    # Pre-load
    real_images, spatial_maps_raw, morphologies, stems = [], [], [], []
    for img_path, spatial_path, stem in samples:
        real_images.append(Image.open(img_path).convert("RGB"))
        sm = np.load(spatial_path)["map"].astype(np.float32) / 255.0
        spatial_maps_raw.append(sm)
        morphologies.append(torch.tensor(morph_df.loc[stem].values, dtype=torch.float32))
        stems.append(stem)

    # ── Pipeline A ──
    ckpt_a_name = os.path.basename(os.path.normpath(CKPT_A))
    print(f"\n{'='*60}")
    print(f"Loading Pipeline A: {ckpt_a_name}")
    print(f"{'='*60}")
    pipeline_a, film_a = load_pipeline(BASE_MODEL, CKPT_A, device)

    print(f"\nGenerating {len(samples)} images from {ckpt_a_name}...")
    gen_a = generate_all(pipeline_a, spatial_maps_raw, morphologies, device, SEED, COND_SCALE, BATCH_SIZE)
    for idx, img in enumerate(gen_a):
        img.save(OUTPUT_DIR / "ckpt_a" / f"{idx:04d}_{stems[idx]}.png")
    del pipeline_a, film_a
    torch.cuda.empty_cache()

    # ── Pipeline B ──
    ckpt_b_name = os.path.basename(os.path.normpath(CKPT_B))
    print(f"\n{'='*60}")
    print(f"Loading Pipeline B: {ckpt_b_name}")
    print(f"{'='*60}")
    pipeline_b, film_b = load_pipeline(BASE_MODEL, CKPT_B, device)

    print(f"\nGenerating {len(samples)} images from {ckpt_b_name}...")
    gen_b = generate_all(pipeline_b, spatial_maps_raw, morphologies, device, SEED, COND_SCALE, BATCH_SIZE)
    for idx, img in enumerate(gen_b):
        img.save(OUTPUT_DIR / "ckpt_b" / f"{idx:04d}_{stems[idx]}.png")
    del pipeline_b, film_b
    torch.cuda.empty_cache()

    # ── Create Comparison Grids with Labeled Spatial Maps ──
    print(f"\nCreating comparison grids with labeled spatial maps...")
    for idx in tqdm(range(len(samples))):
        # Color-coded spatial map with cell-type legend
        spatial_rgb = spatial_map_to_rgb_with_legend(spatial_maps_raw[idx] * 255.0)
        spatial_rgb = spatial_rgb.resize((512, 512), Image.NEAREST)

        real = real_images[idx].resize((512, 512))
        img_a = gen_a[idx].resize((512, 512))
        img_b = gen_b[idx].resize((512, 512))

        spatial_labeled = add_label(spatial_rgb.copy(), "Spatial Map")
        real_labeled = add_label(real.copy(), "Real H&E")
        a_labeled = add_label(img_a.copy(), ckpt_a_name)
        b_labeled = add_label(img_b.copy(), ckpt_b_name)

        grid = Image.new("RGB", (512 * 4, 512))
        grid.paste(spatial_labeled, (0, 0))
        grid.paste(real_labeled, (512, 0))
        grid.paste(a_labeled, (1024, 0))
        grid.paste(b_labeled, (1536, 0))

        grid.save(OUTPUT_DIR / "grids" / f"{idx:04d}_{stems[idx]}.png")

    print(f"\n{'='*60}")
    print(f"Done! {len(samples)} comparison grids saved to: {OUTPUT_DIR / 'grids'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
