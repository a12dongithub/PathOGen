"""Quick FID calculation: checkpoint-15000 generated vs real images."""
import torch
import torchvision.transforms as T
from torchmetrics.image.fid import FrechetInceptionDistance
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Monkey-patch eigvals for CPU fallback (same as validation_utils.py)
original_eigvals = torch.linalg.eigvals
def eigvals_patched(A):
    if A.is_cuda:
        return original_eigvals(A.cpu()).to(A.device)
    return original_eigvals(A)
torch.linalg.eigvals = eigvals_patched

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

gen_dir = Path("./comparison_output/ckpt_b")
real_dir = Path("../results/512_final_dataset/images")

# Match stems
gen_files = sorted(gen_dir.glob("*.png"))
real_stems = {f.stem: f for f in real_dir.glob("*.png")}

# Build matched pairs from the generated filenames (format: 0000_STEM.png)
pairs = []
for gf in gen_files:
    stem = "_".join(gf.stem.split("_")[1:])  # strip the 0000_ prefix
    if stem in real_stems:
        pairs.append((real_stems[stem], gf))

print(f"Matched {len(pairs)} real-generated pairs")

transform = T.Compose([T.Resize((299, 299)), T.ToTensor()])
fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

batch_size = 16
for i in tqdm(range(0, len(pairs), batch_size), desc="FID features"):
    batch = pairs[i:i+batch_size]
    reals = torch.stack([transform(Image.open(r).convert("RGB")) for r, _ in batch]).to(device)
    gens = torch.stack([transform(Image.open(g).convert("RGB")) for _, g in batch]).to(device)
    fid.update(reals, real=True)
    fid.update(gens, real=False)

score = fid.compute().item()
print(f"\n==> FID (checkpoint-15000): {score:.2f}")
