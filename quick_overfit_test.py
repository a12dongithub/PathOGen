#!/usr/bin/env python
"""
Quick overfitting test for Concat Conditioning architecture.
Trains on a small subset (200 images) for 500 steps to verify:
  1. Loss decreases
  2. FID decreases from baseline
  3. Generated images show spatial correspondence with conditioning maps

Usage (single GPU, ~10 min):
  python quick_overfit_test.py \
      --pretrained_model_name_or_path='Manojb/stable-diffusion-2-1-base' \
      --phase1_unet_checkpoint='./checkpoints/phase1_domain_adapt/checkpoint-30000' \
      --train_data_dir='./data'
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, UNet2DConditionModel
from transformers import AutoTokenizer, CLIPTextModel
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as T
import random

# ── Reuse the same SpatialCondEncoder ──
class SpatialCondEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(5, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 4, 1),
        )
        # Default Kaiming init — no zero-init!

    def forward(self, x):
        return self.net(x)

# ── Dataset (same as train_pathogen.py) ──
class MiniDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, stems, transform, tokenizer):
        self.tile_dir = os.path.join(data_dir, "tiles")
        self.map_dir = os.path.join(data_dir, "spatial_maps")
        self.stems = stems
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        img = cv2.imread(os.path.join(self.tile_dir, f"{stem}.png"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tensor = self.transform(img_pil)

        spatial_map = np.load(os.path.join(self.map_dir, f"{stem}.npz"))['map'].astype(np.float32) / 255.0
        map_tensor = torch.from_numpy(spatial_map).permute(2, 0, 1).float()

        inputs = self.tokenizer("he", max_length=self.tokenizer.model_max_length,
                                padding="max_length", truncation=True, return_tensors="pt")
        return {
            "pixel_values": img_tensor,
            "conditioning_pixel_values": map_tensor,
            "input_ids": inputs.input_ids.squeeze(0),
        }

# ── Generate images with the concat-conditioned UNet ──
@torch.no_grad()
def generate(unet, vae, spatial_encoder, text_encoder, tokenizer,
             noise_scheduler, spatial_maps, device, num_steps=20):
    scheduler = DDIMScheduler.from_config(noise_scheduler.config)
    scheduler.set_timesteps(num_steps, device=device)

    unet.eval()
    spatial_encoder.eval()

    text_inputs = tokenizer(["he"], max_length=tokenizer.model_max_length,
                            padding="max_length", truncation=True, return_tensors="pt")
    text_embeds = text_encoder(text_inputs.input_ids.to(device))[0]

    images_out = []
    for i in range(0, len(spatial_maps), 4):
        batch = spatial_maps[i:i+4]
        bs = len(batch)
        st = torch.stack([torch.from_numpy(sm.astype(np.float32)/255.0).permute(2,0,1) for sm in batch]).to(device)
        sf = spatial_encoder(st)
        te = text_embeds.expand(bs, -1, -1)

        gen = torch.Generator(device=device).manual_seed(42 + i)
        latents = torch.randn(bs, 4, 64, 64, generator=gen, device=device)
        latents = latents * scheduler.init_noise_sigma

        for t in scheduler.timesteps:
            inp = scheduler.scale_model_input(latents, t)
            inp = torch.cat([inp, sf], dim=1)
            with torch.autocast("cuda"):
                pred = unet(inp, t, encoder_hidden_states=te, return_dict=False)[0]
            latents = scheduler.step(pred, t, latents, return_dict=False)[0]

        decoded = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
        decoded = (decoded / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()
        for d in decoded:
            images_out.append(Image.fromarray((d * 255).astype(np.uint8)))

    unet.train()
    spatial_encoder.train()
    return images_out

# ── Quick FID on a small set ──
def quick_fid(real_pils, gen_pils, device):
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    tf = T.Compose([T.Resize((299, 299)), T.ToTensor()])
    for img in real_pils:
        fid.update(tf(img).unsqueeze(0).to(device), real=True)
    for img in gen_pils:
        fid.update(tf(img).unsqueeze(0).to(device), real=False)
    return fid.compute().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--phase1_unet_checkpoint", type=str, required=True)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=200)
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output_dir", type=str, default="./overfit_test_output")
    args = parser.parse_args()

    device = torch.device("cuda")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("QUICK OVERFIT TEST: Concat Conditioning")
    print("=" * 60)

    # ── Load models ──
    print("[1/5] Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # Load Phase 1 weights
    phase1_unet = UNet2DConditionModel.from_pretrained(args.phase1_unet_checkpoint, subfolder="unet")
    unet.load_state_dict(phase1_unet.state_dict())
    del phase1_unet
    print("  Loaded Phase 1 UNet weights.")

    # Create spatial encoder
    spatial_encoder = SpatialCondEncoder()
    print(f"  SpatialCondEncoder: {sum(p.numel() for p in spatial_encoder.parameters()):,} params")

    # Expand conv_in from 4 → 8 channels
    old_conv_in = unet.conv_in
    new_conv_in = nn.Conv2d(8, old_conv_in.out_channels,
                            kernel_size=old_conv_in.kernel_size,
                            stride=old_conv_in.stride,
                            padding=old_conv_in.padding)
    with torch.no_grad():
        new_conv_in.weight[:, :4] = old_conv_in.weight
        new_conv_in.weight[:, 4:] = 0.0
        new_conv_in.bias.copy_(old_conv_in.bias)
    unet.conv_in = new_conv_in
    unet.config['in_channels'] = 8
    print("  Expanded UNet conv_in: 4 → 8 channels")

    unet.to(device)
    spatial_encoder.to(device)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    vae.eval()

    # ── Prepare small dataset ──
    print(f"[2/5] Preparing {args.num_images}-image subset...")
    map_dir = os.path.join(args.train_data_dir, "spatial_maps")
    all_stems = sorted([os.path.splitext(f)[0] for f in os.listdir(map_dir) if f.endswith(".npz")])
    rng = random.Random(42)
    selected_stems = rng.sample(all_stems, min(args.num_images, len(all_stems)))
    print(f"  Selected {len(selected_stems)} stems from {len(all_stems)} total.")

    transform = transforms.Compose([
        transforms.Resize(512), transforms.CenterCrop(512),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
    ])
    dataset = MiniDataset(args.train_data_dir, selected_stems, transform, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    # Load real images + spatial maps for evaluation
    eval_reals = []
    eval_spatials = []
    for stem in selected_stems[:50]:  # Evaluate on first 50
        img = Image.open(os.path.join(args.train_data_dir, "tiles", f"{stem}.png")).convert("RGB")
        sm = np.load(os.path.join(map_dir, f"{stem}.npz"))['map']
        eval_reals.append(img)
        eval_spatials.append(sm)

    # ── Optimizer ──
    print("[3/5] Setting up optimizer...")
    params = [
        {"params": list(unet.parameters()), "lr": args.lr},
        {"params": list(spatial_encoder.parameters()), "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    # ── Baseline FID (before any training) ──
    print("[4/5] Computing baseline FID (step 0)...")
    gen_imgs = generate(unet, vae, spatial_encoder, text_encoder, tokenizer,
                        noise_scheduler, eval_spatials, device)
    baseline_fid = quick_fid(eval_reals, gen_imgs, device)
    print(f"  Baseline FID: {baseline_fid:.2f}")
    # Save baseline grids
    for j in range(min(8, len(gen_imgs))):
        gen_imgs[j].save(os.path.join(args.output_dir, f"step0_gen_{j}.png"))
        eval_reals[j].save(os.path.join(args.output_dir, f"step0_real_{j}.png"))

    # ── Training loop ──
    print(f"[5/5] Training for {args.num_steps} steps...")
    unet.train()
    spatial_encoder.train()
    global_step = 0
    losses = []

    while global_step < args.num_steps:
        for batch in dataloader:
            if global_step >= args.num_steps:
                break

            pixel_values = batch["pixel_values"].to(device)
            cond = batch["conditioning_pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            # Encode
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                encoder_hidden_states = text_encoder(input_ids)[0]

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Forward (mixed precision)
            with torch.autocast("cuda"):
                spatial_features = spatial_encoder(cond)
                unet_input = torch.cat([noisy_latents, spatial_features], dim=1)
                model_pred = unet(unet_input, timesteps, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
                loss = F.mse_loss(model_pred.float(), noise.float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(unet.parameters()) + list(spatial_encoder.parameters()), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            losses.append(loss.item())

            if global_step % 10 == 0:
                avg_loss = sum(losses[-10:]) / len(losses[-10:])
                print(f"  Step {global_step:4d} | Loss: {avg_loss:.6f}")

            # Periodic evaluation
            if global_step % args.eval_every == 0:
                gen_imgs = generate(unet, vae, spatial_encoder, text_encoder, tokenizer,
                                    noise_scheduler, eval_spatials, device)
                fid = quick_fid(eval_reals, gen_imgs, device)
                print(f"  ────── Step {global_step} FID: {fid:.2f} (baseline: {baseline_fid:.2f}) ──────")

                # Save sample grids
                for j in range(min(8, len(gen_imgs))):
                    gen_imgs[j].save(os.path.join(args.output_dir, f"step{global_step}_gen_{j}.png"))

                unet.train()
                spatial_encoder.train()

    print("\n" + "=" * 60)
    print("DONE! Check outputs in:", args.output_dir)
    print(f"Baseline FID: {baseline_fid:.2f}")
    print(f"Final FID:    {fid:.2f}")
    print("Look at the generated images to see if spatial correspondence emerged.")
    print("=" * 60)


if __name__ == "__main__":
    main()
