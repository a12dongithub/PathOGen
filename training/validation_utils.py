import os
import random
import torch
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as T
from diffusers import DDIMScheduler
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
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


def _collect_validation_pairs(args, accelerator):
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

    all_pairs = []
    for file in val_tiles_dir.glob("*.png"):
        stem = file.stem
        spatial_path = val_spatial_dir / f"{stem}.npz"
        if spatial_path.exists():
            all_pairs.append((file, spatial_path, stem))

    return all_pairs, val_dir


def _load_morph_df(val_dir, args):
    candidate_paths = [
        val_dir / "morphology_stats.parquet",
        val_dir / "morphology_features" / "morphology_stats.parquet",
        Path(args.train_data_dir) / "morphology_features" / "morphology_stats.parquet",
    ]
    for path in candidate_paths:
        if path.exists():
            return pd.read_parquet(path)
    raise FileNotFoundError("Could not find morphology_stats.parquet for validation.")


@torch.no_grad()
def run_phase1_validation(accelerator, args, global_step, num_images=2000, batch_size=16):
    """
    Evaluate the Phase 1 UNet checkpoint on the validation split using unconditional
    generation with the fixed prompt "he". This runs on the main process only so the
    reported FID corresponds to one exact sampled set.
    """
    if not args.phase1_unet_checkpoint:
        accelerator.print("Skipping Phase 1 validation: no phase1 checkpoint provided.")
        return

    all_pairs, _ = _collect_validation_pairs(args, accelerator)
    if len(all_pairs) == 0:
        accelerator.print("Skipping Phase 1 validation: no valid evaluation tiles found.")
        return

    rng = random.Random(42)
    selected_pairs = rng.sample(all_pairs, min(num_images, len(all_pairs)))

    accelerator.print(f"*** Running Phase 1 Validation at Step {global_step} ***")
    accelerator.print(f"Randomly sampled {len(selected_pairs)} validation tiles for Phase 1 FID.")

    phase1_unet = UNet2DConditionModel.from_pretrained(
        args.phase1_unet_checkpoint,
        subfolder="unet",
        torch_dtype=torch.float16,
    )
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=phase1_unet,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    dataset = [(tile_path, stem) for tile_path, _, stem in selected_pairs]
    with accelerator.split_between_processes(dataset) as local_dataset:
        if len(local_dataset) > 0:
            local_tile_paths, local_stems = zip(*local_dataset)
        else:
            local_tile_paths, local_stems = [], []

        local_real_images = [Image.open(tile_path).convert("RGB") for tile_path in local_tile_paths]
        base_seed = args.seed if args.seed is not None else 42
        generator = torch.Generator(device=accelerator.device).manual_seed(base_seed + accelerator.process_index * 100000)
        generated_images = []
        for i in tqdm(range(0, len(local_real_images), batch_size), desc="Phase1 generation", disable=not accelerator.is_local_main_process):
            current_batch_size = min(batch_size, len(local_real_images) - i)
            prompts = ["he"] * current_batch_size
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = pipeline(prompts, num_inference_steps=20, generator=generator).images
            generated_images.extend(outputs)

        if accelerator.is_main_process:
            phase1_out = Path(args.output_dir) / "validation_phase1_images" / f"step_{global_step}"
            phase1_out.mkdir(parents=True, exist_ok=True)
            sample_count = min(100, len(generated_images))
            sampled_indices = random.Random(42 + global_step).sample(range(len(generated_images)), sample_count)
            for idx in sampled_indices:
                image = generated_images[idx]
                image.save(phase1_out / f"{local_stems[idx]}_phase1.png")
                grid = make_image_grid([local_real_images[idx], image], rows=1, cols=2)
                grid.save(phase1_out / f"{local_stems[idx]}_phase1_compare.png")

        fid_score = calculate_fid(local_real_images, generated_images, accelerator)
        if accelerator.is_main_process:
            accelerator.print(f"--> Phase 1 FID at Step {global_step}: {fid_score:.4f}")
            accelerator.log({"val/phase1_fid": fid_score}, step=global_step)

    del pipeline
    del phase1_unet
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()


@torch.no_grad()
def generate_concat_conditioned(unet, vae, spatial_encoder, text_encoder, tokenizer,
                                 noise_scheduler, spatial_maps, morph_vectors, device, weight_dtype,
                                 num_inference_steps=20, seed=42):
    """
    Manual denoising loop for concat-conditioned UNet (no ControlNet pipeline).
    Uses DDIM scheduler with explicit SD 2.1 parameters.
    
    Args:
        spatial_maps: list of numpy arrays (H, W, 5), values 0-255
        morph_vectors: list of 16D morphology vectors aligned with the spatial maps.
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
    spatial_encoder.eval()
    
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
        
        morph_batch = torch.stack([
            mv if isinstance(mv, torch.Tensor) else torch.tensor(mv, dtype=torch.float32)
            for mv in morph_vectors[i:i+batch_size]
        ]).to(device, dtype=weight_dtype)  # (bs, 16)

        spatial_features = spatial_encoder(spatial_tensor)  # (bs, 4, 64, 64)
        
        # Diagnostic: check if spatial encoder is outputting zeros
        if i == 0:
            feat_abs_mean = spatial_features.abs().mean().item()
            print(f"  [diag] spatial_encoder output |mean|: {feat_abs_mean:.8f}")
            print(f"  [diag] morph16 (first tile): {morph_batch[0].tolist()}")
        
        # Expand text embeddings for the batch
        batch_text_embeds = text_embeds.expand(bs, -1, -1)
        
        # Start from random noise  
        process_seed = seed + i
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            process_seed += torch.distributed.get_rank() * 100000
        generator = torch.Generator(device=device).manual_seed(process_seed)
        latents = torch.randn(bs, 4, 64, 64, generator=generator, device=device, dtype=weight_dtype)
        latents = latents * scheduler.init_noise_sigma

        for module in unet.modules():
            if hasattr(module, "film_mlp"):
                module.current_morph16 = morph_batch
        
        # Denoising loop
        for t in scheduler.timesteps:
            latent_model_input = scheduler.scale_model_input(latents, t)
            
            # Concat spatial features with noisy latents
            unet_input = torch.cat([latent_model_input, spatial_features], dim=1)  # (bs, 8, 64, 64)
            
            with torch.autocast("cuda", dtype=weight_dtype):
                noise_pred = unet(
                    unet_input, t,
                    encoder_hidden_states=batch_text_embeds,
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
    for module in unet.modules():
        if hasattr(module, "film_mlp"):
            module.current_morph16 = None
    unet.train()
    spatial_encoder.train()
    
    return generated


def run_phase2_validation(accelerator, unet, vae, spatial_encoder,
                          text_encoder, tokenizer, noise_scheduler,
                          args, global_step, weight_dtype):
    """
    Runs conditional concat-based generation for Phase 2, computes FID, and creates visual comparison grids.
    Uses random sampling of validation images for unbiased FID estimation.
    """
    accelerator.print(f"*** Running Phase 2 Validation at Step {global_step} ***")

    all_pairs, val_dir = _collect_validation_pairs(args, accelerator)
    if len(all_pairs) == 0:
        accelerator.print("Skipping validation: No valid pairs found.")
        return

    morph_df = _load_morph_df(val_dir, args)

    valid_pairs = [pair for pair in all_pairs if pair[2] in morph_df.index]
    if len(valid_pairs) == 0:
        accelerator.print("Skipping validation: No valid morphology rows found for selected tiles.")
        return

    rng = random.Random(42)
    num_samples = min(2000, len(valid_pairs))
    selected_pairs = rng.sample(valid_pairs, num_samples)
    accelerator.print(f"Randomly sampled {num_samples} / {len(valid_pairs)} pairs for FID evaluation.")
    
    accelerator.print(f"Generating {len(selected_pairs)} conditional H&E tiles...")

    dataset = selected_pairs
    with accelerator.split_between_processes(dataset) as local_dataset:
        if len(local_dataset) > 0:
            local_tile_paths, local_spatial_paths, local_stems = zip(*local_dataset)
        else:
            local_tile_paths, local_spatial_paths, local_stems = [], [], []

        local_real_images = [Image.open(tile_path).convert("RGB") for tile_path in local_tile_paths]
        local_spatial_maps = [np.load(spatial_path)["map"] for spatial_path in local_spatial_paths]
        local_morph_vectors = [torch.tensor(morph_df.loc[stem].values, dtype=torch.float32) for stem in local_stems]

        generated_images = generate_concat_conditioned(
            unet, vae, spatial_encoder, text_encoder, tokenizer,
            noise_scheduler, local_spatial_maps, local_morph_vectors, accelerator.device, weight_dtype,
            num_inference_steps=20, seed=args.seed if args.seed else 42,
        )

        if accelerator.is_main_process:
            val_out = Path(args.output_dir) / "validation_images" / f"step_{global_step}"
            val_out.mkdir(parents=True, exist_ok=True)
            sample_count = min(100, len(generated_images))
            sampled_indices = random.Random(42 + global_step).sample(range(len(generated_images)), sample_count)
            for j in sampled_indices:
                out = generated_images[j]
                grid = make_image_grid([local_real_images[j], out], rows=1, cols=2)
                grid.save(val_out / f"{local_stems[j]}_compare.png")

        fid_score = calculate_fid(local_real_images, generated_images, accelerator)
        if accelerator.is_main_process:
            accelerator.print(f"--> Phase 2 FID at Step {global_step}: {fid_score:.4f}")
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

    accelerator.wait_for_everyone()
