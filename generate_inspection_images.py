import os
import torch
from pathlib import Path
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

def main():
    checkpoint_dir = "./checkpoint-30000/checkpoint-30000"
    base_model_id = "Manojb/stable-diffusion-2-1-base"
    output_dir = Path("./inspection_images_step30k")
    
    if not os.path.exists(checkpoint_dir):
        print(f"Error: {checkpoint_dir} does not exist.")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    num_images = 100
    batch_size = 10  # Fit comfortably on GPU
    
    print(f"Generating {num_images} images...")
    generator = torch.Generator("cuda").manual_seed(42)
    
    generated_count = 0
    with torch.autocast("cuda"):
        for i in tqdm(range(0, num_images, batch_size)):
            current_batch = min(batch_size, num_images - i)
            prompts = ["he"] * current_batch
            
            images = pipeline(prompts, num_inference_steps=30, generator=generator).images
            
            for img in images:
                img.save(output_dir / f"generated_he_{generated_count:03d}.png")
                generated_count += 1
                
    print(f"Successfully generated {generated_count} images to {output_dir}")

if __name__ == "__main__":
    main()
