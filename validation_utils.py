import os
import random
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as T
from diffusers import DDIMScheduler
from diffusers.utils import make_image_grid

# === Monkey Patch PyTorch CUDA Linalg to Bypass Missing Nightly Library ===
original_eigvals = torch.linalg.eigvals
def eigvals_patched(A):
    if A.is_cuda:
        return original_eigvals(A.cpu()).to(A.device)
    return original_eigvals(A)
torch.linalg.eigvals = eigvals_patched
# =========================================================================

def calculate_fid(real_images, generated_images, accelerator):
    """
    Computes Frechet Inception Distance between two lists of PIL images using torchmetrics.
    """
    device = accelerator.device
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    transform = T.Compose([
        T.Resize((299, 299)),
        T.ToTensor(),
    ])
    
    batch_size = 32
    for i in range(0, len(real_images), batch_size):
        real_batch = torch.stack([transform(img) for img in real_images[i:i+batch_size]]).to(device)
        fid.update(real_batch, real=True)
        
    for i in range(0, len(generated_images), batch_size):
        gen_batch = torch.stack([transform(img) for img in generated_images[i:i+batch_size]]).to(device)
        fid.update(gen_batch, real=False)
        
    return fid.compute().item()


@torch.no_grad()
def generate_controlnet_conditioned(unet, vae, controlnet, text_encoder, tokenizer,
                                     noise_scheduler, spatial_maps, device, weight_dtype,
                                     num_inference_steps=20, seed=42):
    """
    Manual denoising loop using ControlNet for spatial conditioning.
    ControlNet processes spatial maps at pixel resolution and injects features
    at multiple UNet levels via residual connections.
    UNet conv_in stays at 4 channels (untouched).
    
    Args:
        spatial_maps: list of numpy arrays (H, W, 5), values 0-255
    Returns:
        list of PIL images
    """
    # Create DDIM scheduler with explicit SD 2.1 parameters.
    scheduler = DDIMScheduler(
        beta_start=noise_scheduler.config.beta_start,
        beta_end=noise_scheduler.config.beta_end,
        beta_schedule=noise_scheduler.config.beta_schedule,
        num_train_timesteps=noise_scheduler.config.num_train_timesteps,
        prediction_type=noise_scheduler.config.prediction_type,
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
        timestep_spacing="leading",
    )
    scheduler.set_timesteps(num_inference_steps, device=device)
    
    # Switch models to eval mode for deterministic inference
    unet.eval()
    controlnet.eval()
    
    # Encode text (constant "he" prompt)
    text_inputs = tokenizer(
        ["he"], max_length=tokenizer.model_max_length,
        padding="max_length", truncation=True, return_tensors="pt"
    )
    text_embeds = text_encoder(text_inputs.input_ids.to(device), return_dict=False)[0]
    
    generated = []
    batch_size = 4  # Keep small for V100 32GB memory
    num_batches = (len(spatial_maps) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(spatial_maps), batch_size), total=num_batches, desc="Generating"):
        current_batch = spatial_maps[i:i+batch_size]
        bs = len(current_batch)
        
        # Prepare spatial conditioning (normalize 0-255 -> 0-1)
        spatial_tensor = torch.stack([
            torch.from_numpy(sm.astype(np.float32) / 255.0).permute(2, 0, 1)
            for sm in current_batch
        ]).to(device, dtype=weight_dtype)  # (bs, 5, 512, 512)
        
        # Expand text embeddings for the batch
        batch_text_embeds = text_embeds.expand(bs, -1, -1)
        
        # Start from random noise  
        generator = torch.Generator(device=device).manual_seed(seed + i)
        latents = torch.randn(bs, 4, 64, 64, generator=generator, device=device, dtype=weight_dtype)
        latents = latents * scheduler.init_noise_sigma
        
        # Denoising loop
        for t in scheduler.timesteps:
            latent_model_input = scheduler.scale_model_input(latents, t)
            
            with torch.autocast("cuda", dtype=weight_dtype):
                # ControlNet: spatial maps at pixel resolution → residuals
                down_block_res_samples, mid_block_res_sample = controlnet(
                    latent_model_input, t,
                    encoder_hidden_states=batch_text_embeds,
                    controlnet_cond=spatial_tensor,
                    return_dict=False,
                )
                
                # UNet: standard 4-channel input + ControlNet residuals
                noise_pred = unet(
                    latent_model_input, t,  # 4-channel, NO concat
                    encoder_hidden_states=batch_text_embeds,
                    down_block_additional_residuals=[s.clone() for s in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample.clone(),
                    return_dict=False,
                )[0]
            
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        # Decode latents to images
        latents_for_decode = latents / vae.config.scaling_factor
        images = vae.decode(latents_for_decode.float(), return_dict=False)[0]
        
        # Convert to PIL (VAE output is in [-1, 1])
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        for img_np in images:
            pil_img = Image.fromarray((img_np * 255).astype(np.uint8))
            generated.append(pil_img)
    
    # Switch models back to train mode
    unet.train()
    controlnet.train()
    
    return generated


def run_phase2_validation(accelerator, unet, vae, controlnet,
                          text_encoder, tokenizer, noise_scheduler,
                          args, global_step, weight_dtype):
    """
    Runs ControlNet-conditioned generation for Phase 2v2, computes FID, and creates visual comparison grids.
    Uses random sampling of validation images for unbiased FID estimation.
    """
    accelerator.print(f"*** Running Phase 2 Validation at Step {global_step} ***")
    
    val_dir = Path(args.train_data_dir).parent / "data_val"
    val_tiles_dir = val_dir / "tiles"
    val_spatial_dir = val_dir / "spatial_maps"
    
    if not val_tiles_dir.exists():
        accelerator.print(f"Validation dir {val_tiles_dir} not found. Falling back to training data subset.")
        val_dir = Path(args.train_data_dir)
        val_tiles_dir = val_dir / "tiles"
        if not val_tiles_dir.exists():
            val_tiles_dir = val_dir / "images"
        val_spatial_dir = val_dir / "spatial_maps"
    
    # Collect all valid (tile + spatial_map) pairs
    all_pairs = []
    for file in val_tiles_dir.glob("*.png"):
        stem = file.stem
        spatial_path = val_spatial_dir / f"{stem}.npz"
        if spatial_path.exists():
            all_pairs.append((file, spatial_path, stem))
    
    if len(all_pairs) == 0:
        accelerator.print("Skipping validation: No valid pairs found.")
        return
    
    # Randomly sample 2000 pairs (seeded for reproducibility across runs)
    rng = random.Random(42)
    num_samples = min(2000, len(all_pairs))
    selected_pairs = rng.sample(all_pairs, num_samples)
    accelerator.print(f"Randomly sampled {num_samples} / {len(all_pairs)} pairs for FID evaluation.")
    
    real_images = []
    spatial_maps = []
    stems = []
    for tile_path, spatial_path, stem in selected_pairs:
        real_images.append(Image.open(tile_path).convert("RGB"))
        spatial_data = np.load(spatial_path)
        spatial_maps.append(spatial_data['map'])
        stems.append(stem)

    accelerator.print(f"Generating {len(real_images)} conditional H&E tiles...")
    
    dataset = list(zip(real_images, spatial_maps, stems))
    with accelerator.split_between_processes(dataset) as local_dataset:
        if len(local_dataset) > 0:
            local_reals, local_spatials, local_stems = zip(*local_dataset)
        else:
            local_reals, local_spatials, local_stems = [], [], []
        
        # Generate images using ControlNet-conditioned denoising loop
        generated_images = generate_controlnet_conditioned(
            unet, vae, controlnet, text_encoder, tokenizer,
            noise_scheduler, list(local_spatials), accelerator.device, weight_dtype,
            num_inference_steps=20, seed=args.seed if args.seed else 42,
        )
        
        # Save comparison grids (main process only, first 100)
        if accelerator.is_main_process:
            val_out = Path(args.output_dir) / "validation_images" / f"step_{global_step}"
            val_out.mkdir(parents=True, exist_ok=True)
            for j, out in enumerate(generated_images):
                if j < 100:
                    grid = make_image_grid([local_reals[j], out], rows=1, cols=2)
                    grid.save(val_out / f"{local_stems[j]}_compare.png")

        # Calculate FID
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.print("Computing FID across all GPUs...")
            
        fid_score = calculate_fid(list(local_reals), generated_images, accelerator)
        
        if accelerator.is_main_process:
            accelerator.print(f"--> FID Score at Step {global_step}: {fid_score:.4f}")
            accelerator.log({"val/fid": fid_score}, step=global_step)

    # Log image grids to Tensorboard
    if accelerator.is_main_process:
        try:
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    tensor_grids = []
                    val_out = Path(args.output_dir) / "validation_images" / f"step_{global_step}"
                    for grid_file in sorted(val_out.glob("*_compare.png"))[:16]:
                        img = Image.open(grid_file).convert("RGB")
                        t = T.ToTensor()(img)
                        tensor_grids.append(t)
                    if tensor_grids:
                        batch_tensors = torch.stack(tensor_grids)
                        tracker.writer.add_images("val/real_vs_generated", batch_tensors, global_step)
        except Exception as e:
            accelerator.print(f"Could not log image grids to TB: {e}")
