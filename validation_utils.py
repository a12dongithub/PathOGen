import os
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as T
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline
from diffusers.utils import make_image_grid

# === Monkey Patch PyTorch CUDA Linalg to Bypass Missing Nightly Library ===
original_eigvals = torch.linalg.eigvals
def eigvals_patched(A):
    if A.is_cuda:
        # Compute eigenvalues on CPU to dodge libtorch_cuda_linalg.so error
        return original_eigvals(A.cpu()).to(A.device)
    return original_eigvals(A)
torch.linalg.eigvals = eigvals_patched
# =========================================================================

def calculate_fid(real_images, generated_images, accelerator):
    """
    Computes Frechet Inception Distance between two lists of PIL images using torchmetrics.
    Extracts features on GPU. The CPU eigen-patch safely dodges the PyTorch nightly CUDA crash.
    """
    device = accelerator.device
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    # Preprocessing: FID expects Float tensors [0, 1] (because we set normalize=True)
    transform = T.Compose([
        T.Resize((299, 299)),
        T.ToTensor(),
    ])
    
    # Process in chunks to avoid OOM
    batch_size = 32
    for i in range(0, len(real_images), batch_size):
        real_batch = torch.stack([transform(img) for img in real_images[i:i+batch_size]]).to(device)
        fid.update(real_batch, real=True)
        
    for i in range(0, len(generated_images), batch_size):
        gen_batch = torch.stack([transform(img) for img in generated_images[i:i+batch_size]]).to(device)
        fid.update(gen_batch, real=False)
        
    # Natively synchronizes across all distributed chunks and computes score!
    return fid.compute().item()

def run_phase1_validation(accelerator, pipeline, args, global_step):
    """
    Runs unconditional H&E generation for Phase 1 and computes FID against the val set.
    """
    accelerator.print(f"*** Running Phase 1 Validation at Step {global_step} ***")
    
    val_dir = Path(args.train_data_dir).parent / "data_val"
    val_tiles_dir = val_dir / "tiles"
    
    if not val_tiles_dir.exists():
        accelerator.print(f"Skipping validation: {val_tiles_dir} not found.")
        return
        
    real_images = []
    for file in sorted(val_tiles_dir.glob("*.png")):
        real_images.append(Image.open(file).convert("RGB"))
        
    if len(real_images) == 0:
        accelerator.print("Skipping validation: Empty validation directory.")
        return

    # Generate unconditional images (using the static "he" prompt)
    accelerator.print(f"Generating {len(real_images)} unconditional H&E tiles...")
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed if args.seed else 42)
    pipeline.set_progress_bar_config(disable=True)
    
    generated_images = []
    weight_dtype = pipeline.unet.dtype
    
    with accelerator.split_between_processes(real_images) as local_real_images:
        batch_size = 16
        for i in tqdm(range(0, len(local_real_images), batch_size), disable=not accelerator.is_local_main_process):
            current_batch_size = min(batch_size, len(local_real_images) - i)
            prompts = ["he"] * current_batch_size
            
            with torch.autocast("cuda", dtype=weight_dtype):
                out = pipeline(prompts, num_inference_steps=20, generator=generator).images
            generated_images.extend(out)
        
        # Calculate FID across all distributed processes natively using torchmetrics
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.print("Computing FID across all GPUs...")
            
        fid_score = calculate_fid(local_real_images, generated_images, accelerator)
        
        # Log to tracking platform (WandB / Tensorboard) on Main Process
        if accelerator.is_main_process:
            accelerator.print(f"--> FID Score at Step {global_step}: {fid_score:.4f}")
            accelerator.log({"val/fid": fid_score}, step=global_step)
    
def run_phase2_validation(accelerator, pipeline, args, global_step):
    """
    Runs conditional ControlNet+FiLM generation for Phase 2, computes FID, and creates visual comparison grids.
    """
    accelerator.print(f"*** Running Phase 2 Validation at Step {global_step} ***")
    
    val_dir = Path(args.train_data_dir).parent / "data_val"
    val_tiles_dir = val_dir / "tiles"
    val_spatial_dir = val_dir / "spatial_maps"
    val_morph_path = val_dir / "morphology_stats.parquet"
    
    if not val_tiles_dir.exists():
        accelerator.print(f"Skipping validation: {val_tiles_dir} not found.")
        return
        
    real_images = []
    spatial_maps = []
    morphologies = []
    stems = []
    
    if not val_morph_path.exists():
        accelerator.print(f"Skipping validation: {val_morph_path} not found.")
        return
        
    import pandas as pd
    morph_df = pd.read_parquet(val_morph_path)
    
    # Load evaluation batch
    for file in sorted(val_tiles_dir.glob("*.png")):
        stem = file.stem
        spatial_path = val_spatial_dir / f"{stem}.npz"
        
        if not spatial_path.exists() or stem not in morph_df.index:
            continue
            
        real_images.append(Image.open(file).convert("RGB"))
        
        # Load compressed .npz map
        spatial_data = np.load(spatial_path)
        spatial_maps.append(spatial_data['map'])
        
        # Load morphology features from parquet
        morph_row = torch.tensor(morph_df.loc[stem].values, dtype=torch.float32)
        morphologies.append(morph_row)
        
        stems.append(stem)

    if len(real_images) == 0:
        accelerator.print("Skipping validation: Incomplete triplets in validation directory.")
        return

    accelerator.print(f"Generating {len(real_images)} conditional H&E tiles...")
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed if args.seed else 42)
    pipeline.set_progress_bar_config(disable=True)
    
    generated_images = []
    visualization_grids = []
    weight_dtype = pipeline.unet.dtype
    
    dataset = list(zip(real_images, spatial_maps, morphologies, stems))
    with accelerator.split_between_processes(dataset) as local_dataset:
        if len(local_dataset) > 0:
            local_reals, local_spatials, local_morphs, local_stems = zip(*local_dataset)
        else:
            local_reals, local_spatials, local_morphs, local_stems = [], [], [], []
            
        batch_size = 16
        for i in tqdm(range(0, len(local_reals), batch_size), disable=not accelerator.is_local_main_process):
            current_batch_size = min(batch_size, len(local_reals) - i)
            
            # Batch slices
            spatial_batch = local_spatials[i : i + current_batch_size]
            morph_batch = local_morphs[i : i + current_batch_size]
            prompts = ["he"] * current_batch_size
            
            # Prepare batched tensors
            spatial_tensor = torch.stack([
                torch.from_numpy(sm).float().permute(2, 0, 1)
            for sm in spatial_batch]).to(accelerator.device, dtype=weight_dtype)
            
            morph_tensor = torch.stack(morph_batch).to(accelerator.device, dtype=weight_dtype)
            
            # Stable Diffusion CFG runs batched unconditional and conditional passes.
            # So the UNet actually processes (batch_size * 2) examples. 
            # We must explicitly attach the duplicated morphology tensor to the unet blocks.
            morph_tensor_dup = torch.cat([morph_tensor, morph_tensor], dim=0)
            
            for module in pipeline.unet.modules():
                if hasattr(module, "current_morph16"):
                    module.current_morph16 = morph_tensor_dup
            
            # Forward pass through pipeline
            with torch.autocast("cuda", dtype=weight_dtype):
                outputs = pipeline(
                    prompt=prompts, 
                    image=spatial_tensor,
                    controlnet_conditioning_scale=0.5,
                    num_inference_steps=20, 
                    generator=generator
                ).images
                
            generated_images.extend(outputs)
            
            # Only save the side-by-side grids on the main process to prevent duplicates
            if accelerator.is_main_process:
                for j, out in enumerate(outputs):
                    global_idx = i + j
                    if global_idx < 100:
                        grid = make_image_grid([local_reals[global_idx], out], rows=1, cols=2)
                        visualization_grids.append(grid)
                        
                        # Optionally save locally
                        val_out = Path(args.output_dir) / "validation_images" / f"step_{global_step}"
                        val_out.mkdir(parents=True, exist_ok=True)
                        grid.save(val_out / f"{local_stems[global_idx]}_compare.png")

        # Calculate FID collaboratively
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            accelerator.print("Computing FID across all GPUs...")
            
        fid_score = calculate_fid(list(local_reals), generated_images, accelerator)
        
        # Log to Tensorboard ONLY on main process
        if accelerator.is_main_process:
            accelerator.print(f"--> FID Score at Step {global_step}: {fid_score:.4f}")
            accelerator.log({"val/fid": fid_score}, step=global_step)
    # 3. Log Image Grids to Tensorboard
    # Tensorboard format: Tensor of shape (N, C, H, W) where N is images
    tracker = accelerator.get_tracker("tensorboard")
    if tracker is not None:
        try:
            # Convert PIL grid back to tensors for logging
            tensor_grids = []
            for img in visualization_grids:
                t = T.ToTensor()(img)
                tensor_grids.append(t)
            batch_tensors = torch.stack(tensor_grids)
            tracker.writer.add_images("val/real_vs_generated", batch_tensors, global_step)
        except Exception as e:
            accelerator.print(f"Could not log image grids to TB: {e}")
