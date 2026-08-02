import os
import random
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as T

# === Monkey Patch PyTorch CUDA Linalg to Bypass Missing Nightly Library ===
original_eigvals = torch.linalg.eigvals
def eigvals_patched(A):
    if A.is_cuda:
        return original_eigvals(A.cpu()).to(A.device)
    return original_eigvals(A)
torch.linalg.eigvals = eigvals_patched
# =========================================================================

def main():
    checkpoint_dir = "./checkpoint-30000/checkpoint-30000"
    base_model_id = "Manojb/stable-diffusion-2-1-base"
    output_dir = Path("./generated_2000_step30k")
    real_images_dir = Path(r"c:\Users\samar\Documents\CVPR\Interpretation\CellVit\CellViT-plus-plus-main\results\512_final_dataset\images")
    
    # 1. Generate 2000 Images
    output_dir.mkdir(parents=True, exist_ok=True)
    num_to_generate = 2000
    batch_size = 16 # Optimized batch size
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    existing_generated = list(output_dir.glob("*.png"))
    if len(existing_generated) >= num_to_generate:
        print(f"Already found {len(existing_generated)} generated images. Skipping generation.")
    else:
        if not os.path.exists(checkpoint_dir):
            print(f"Error: {checkpoint_dir} does not exist.")
            return
            
        print(f"Loading UNet from {checkpoint_dir}/unet...")
        unet = UNet2DConditionModel.from_pretrained(f"{checkpoint_dir}/unet", torch_dtype=torch.float16)
        
        print(f"Loading Pipeline with Base Model {base_model_id}...")
        pipeline = StableDiffusionPipeline.from_pretrained(
            base_model_id,
            unet=unet,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        pipeline.to("cuda")
        pipeline.set_progress_bar_config(disable=True)
        
        print(f"Generating {num_to_generate - len(existing_generated)} new images...")
        generator = torch.Generator("cuda").manual_seed(42)
        
        generated_count = len(existing_generated)
        with torch.autocast("cuda"):
            for i in tqdm(range(generated_count, num_to_generate, batch_size)):
                current_batch = min(batch_size, num_to_generate - i)
                prompts = ["he"] * current_batch
                
                # 20 inference steps is standard/fast for FID proxy without losing much quality
                images = pipeline(prompts, num_inference_steps=20, generator=generator).images
                
                for img in images:
                    img.save(output_dir / f"generated_he_{generated_count:04d}.png")
                    generated_count += 1
                    
        print(f"Successfully generated 2000 images to {output_dir}")
        
        # Free up VRAM
        del pipeline
        del unet
        torch.cuda.empty_cache()

    # 2. Calculate FID
    print("\nCalculating FID Score (Real vs. Generated)")
    
    all_real_images = list(real_images_dir.glob("*.png"))
    print(f"Found {len(all_real_images)} total real images in dataset.")
    random.seed(42)  # Seed for consistency
    sampled_real_paths = random.sample(all_real_images, 2000)
    
    all_gen_images = list(output_dir.glob("*.png"))[:2000]
    
    if len(all_gen_images) < 2000:
        print("Not enough generated images for FID calculation.")
        return
        
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    transform = T.Compose([
        T.Resize((299, 299)),
        T.ToTensor(),
    ])
    
    fid_batch_size = 32
    print("Processing Real Images...")
    for i in tqdm(range(0, 2000, fid_batch_size)):
        batch_paths = sampled_real_paths[i:i+fid_batch_size]
        batch_tensors = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            batch_tensors.append(transform(img))
        batch_tensors = torch.stack(batch_tensors).to(device)
        fid.update(batch_tensors, real=True)
        
    print("Processing Generated Images...")
    for i in tqdm(range(0, 2000, fid_batch_size)):
        batch_paths = all_gen_images[i:i+fid_batch_size]
        batch_tensors = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            batch_tensors.append(transform(img))
        batch_tensors = torch.stack(batch_tensors).to(device)
        fid.update(batch_tensors, real=False)
        
    print("Computing generative FID score...")
    score = fid.compute().item()
    print(f"\n=========================================")
    print(f"-> Generative FID Score (Model 30k vs. Real): {score:.4f}")
    print(f"=========================================\n")

if __name__ == "__main__":
    main()
