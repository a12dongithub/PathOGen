import os
import random
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
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
    print("Calculating Baseline FID (Real vs. Real)")
    image_dir = Path(r"c:\Users\samar\Documents\CVPR\Interpretation\CellVit\CellViT-plus-plus-main\results\512_final_dataset\images")
    
    # Get all PNG images
    all_images = list(image_dir.glob("*.png"))
    print(f"Found {len(all_images)} total images in dataset.")
    
    if len(all_images) < 4000:
        print("Not enough images! Need at least 4000.")
        return
        
    # Shuffle and sample 4000
    random.seed(42)
    sampled_imgs = random.sample(all_images, 4000)
    
    set_a_paths = sampled_imgs[:2000]
    set_b_paths = sampled_imgs[2000:]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    transform = T.Compose([
        T.Resize((299, 299)),
        T.ToTensor(),
    ])
    
    batch_size = 32
    print("Processing Set A (First 2000 Real Images)...")
    for i in tqdm(range(0, 2000, batch_size)):
        batch_paths = set_a_paths[i:i+batch_size]
        batch_tensors = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            batch_tensors.append(transform(img))
        batch_tensors = torch.stack(batch_tensors).to(device)
        fid.update(batch_tensors, real=True)
        
    print("Processing Set B (Second 2000 Real Images as 'Fake')...")
    for i in tqdm(range(0, 2000, batch_size)):
        batch_paths = set_b_paths[i:i+batch_size]
        batch_tensors = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            batch_tensors.append(transform(img))
        batch_tensors = torch.stack(batch_tensors).to(device)
        fid.update(batch_tensors, real=False)
        
    print("Computing baseline FID score...")
    score = fid.compute().item()
    print(f"\n=========================================")
    print(f"-> Baseline FID Score (Real vs. Real): {score:.4f}")
    print(f"=========================================\n")

if __name__ == "__main__":
    main()
