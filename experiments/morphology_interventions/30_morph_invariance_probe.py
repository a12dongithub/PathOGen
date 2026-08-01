import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import joblib
import time

# Feature extractors
import timm
from torchvision import transforms
from PIL import Image

# Generative model imports
from diffusers import DDIMScheduler, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import sys
sys.path.append(r"src")
from cpathogen.generation.phase2 import SpatialCondEncoder, inject_film_into_unet

# CTransPath import
sys.path.append(r".")
from cpathogen.encoders.ctranspath import ctranspath

# PyTorchMLPClassifier definition (needed for unpickling)
class PyTorchMLPClassifier:
    def __init__(self, input_dim, hidden_layer_sizes=(512, 128), max_iter=100, lr=1e-3, device="cuda"):
        self.input_dim = input_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.lr = lr
        self.device = device
        self.classes_ = None
        self.model = None

    def _build_model(self, num_classes):
        layers = []
        in_d = self.input_dim
        for h in self.hidden_layer_sizes:
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.ReLU())
            in_d = h
        layers.append(nn.Linear(in_d, num_classes))
        return nn.Sequential(*layers).to(self.device)

    def fit(self, X, y):
        # ... dummy implementation ...
        pass

    def predict_proba(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        pred_idx = np.argmax(probs, axis=1)
        return self.classes_[pred_idx]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_DTYPE = torch.float16

# Configuration
SPATIAL_DIR = r"data/processed/generator/spatial_maps"
MORPH_STATS_PATH = r"data/processed/generator/morphology_features/morphology_standardized.parquet"
CKPT_DIR = r"artifacts/runs/legacy_phase2_fid58/checkpoints/checkpoint-30000"
BASE_MODEL = "Manojb/stable-diffusion-2-1-base"

def load_generative_models():
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder", torch_dtype=WEIGHT_DTYPE).to(DEVICE)
    vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae", torch_dtype=WEIGHT_DTYPE).to(DEVICE)
    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")

    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet", torch_dtype=WEIGHT_DTYPE)
    old_conv_in = unet.conv_in
    new_conv_in = torch.nn.Conv2d(8, old_conv_in.out_channels, kernel_size=old_conv_in.kernel_size,
                                    stride=old_conv_in.stride, padding=old_conv_in.padding).to(unet.device, dtype=WEIGHT_DTYPE)
    with torch.no_grad():
        new_conv_in.weight[:, :4] = old_conv_in.weight
        new_conv_in.weight[:, 4:] = 0.0
        new_conv_in.bias.copy_(old_conv_in.bias)
    unet.conv_in = new_conv_in
    unet.config['in_channels'] = 8

    film_mlps = inject_film_into_unet(unet, film_dim=16)
    unet.load_state_dict(UNet2DConditionModel.from_pretrained(CKPT_DIR, subfolder="unet").state_dict(), strict=False)
    unet.to(DEVICE)

    spatial_encoder = SpatialCondEncoder().to(DEVICE, dtype=WEIGHT_DTYPE)
    spatial_encoder.load_state_dict(torch.load(os.path.join(CKPT_DIR, "spatial_encoder.pt"), map_location=DEVICE))
    film_mlps.load_state_dict(torch.load(os.path.join(CKPT_DIR, "film_mlps.pt"), map_location=DEVICE))
    film_mlps.to(DEVICE, dtype=WEIGHT_DTYPE)
    
    vae.eval()
    text_encoder.eval()
    unet.eval()
    spatial_encoder.eval()
    
    return unet, vae, spatial_encoder, text_encoder, tokenizer, noise_scheduler

@torch.no_grad()
def generate_batch(unet, vae, spatial_encoder, text_encoder, tokenizer, noise_scheduler, spatial_maps, morph_vectors, grid_seed=42):
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
    scheduler.set_timesteps(20, device=DEVICE)
    
    text_inputs = tokenizer(["he"], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
    text_embeds = text_encoder(text_inputs.input_ids.to(DEVICE), return_dict=False)[0]
    
    batch_size = len(spatial_maps)
    generator = torch.Generator(device=DEVICE).manual_seed(grid_seed)
    base_latent = torch.randn(1, 4, 64, 64, generator=generator, device=DEVICE, dtype=WEIGHT_DTYPE)
    latents = base_latent.expand(batch_size, -1, -1, -1).clone()
    latents = latents * scheduler.init_noise_sigma
    
    spatial_tensor = torch.stack([
        torch.from_numpy(sm.astype(np.float32) / 255.0).permute(2, 0, 1) for sm in spatial_maps
    ]).to(DEVICE, dtype=WEIGHT_DTYPE)
    
    morph_batch = torch.stack([
        mv if isinstance(mv, torch.Tensor) else torch.tensor(mv, dtype=torch.float32) for mv in morph_vectors
    ]).to(DEVICE, dtype=WEIGHT_DTYPE)

    spatial_features = spatial_encoder(spatial_tensor)
    batch_text_embeds = text_embeds.expand(batch_size, -1, -1)
    
    for module in unet.modules():
        if hasattr(module, "film_mlp"):
            module.current_morph16 = morph_batch
            
    for t in scheduler.timesteps:
        latent_model_input = scheduler.scale_model_input(latents, t)
        unet_input = torch.cat([latent_model_input, spatial_features], dim=1)
        with torch.autocast("cuda", dtype=WEIGHT_DTYPE):
            noise_pred = unet(unet_input, t, encoder_hidden_states=batch_text_embeds, return_dict=False)[0]
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
    latents_for_decode = latents / vae.config.scaling_factor
    images = vae.decode(latents_for_decode, return_dict=False)[0]
    images = (images / 2 + 0.5).clamp(0, 1)
    
    for module in unet.modules():
        if hasattr(module, "film_mlp"):
            module.current_morph16 = None
            
    return images  # Shape: (B, 3, 512, 512) tensor

def load_extractors():
    # UNI-2h
    uni_kwargs = {
        'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
        'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667*2,
        'num_classes': 0, 'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 'dynamic_img_size': True
    }
    uni_model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **uni_kwargs).to(DEVICE).eval()
    
    # CTransPath
    ctr_model = ctranspath()
    ctr_model.head = torch.nn.Identity()
    
    state_dict = torch.load(r"artifacts/models/ctranspath.pth", map_location='cpu')
    if 'model' in state_dict:
        state_dict = state_dict['model']
        
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'downsample' in k:
            parts = k.split('.')
            if parts[0] == 'layers':
                layer_idx = int(parts[1])
                parts[1] = str(layer_idx + 1)
                new_k = '.'.join(parts)
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v
        else:
            new_state_dict[k] = v
            
    ctr_model.load_state_dict(new_state_dict, strict=False)
    ctr_model = ctr_model.to(DEVICE).eval()
    
    # ResNet50
    import torchvision.models as models
    res_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    res_model.fc = nn.Identity()
    res_model = res_model.to(DEVICE).eval()
    
    return uni_model, ctr_model, res_model

def get_transforms():
    uni_t = transforms.Compose([
        transforms.Resize(224),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    ctr_t = transforms.Compose([
        transforms.Resize(224),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    res_t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return uni_t, ctr_t, res_t

@torch.no_grad()
def extract_features(images, uni_model, ctr_model, res_model, uni_t, ctr_t, res_t):
    # images is a (B, 3, 512, 512) tensor in [0, 1]
    uni_in = uni_t(images)
    ctr_in = ctr_t(images)
    res_in = res_t(images)
    
    with torch.amp.autocast('cuda'):
        feat_uni = uni_model(uni_in)
        
        c_out = ctr_model(ctr_in)
        if c_out.dim() == 4:
            c_out = c_out.mean(dim=(1, 2))
        elif c_out.dim() == 3:
            c_out = c_out.mean(dim=1)
        feat_ctr = c_out
        
        r_out = res_model(res_in)
        if r_out.dim() > 2:
            r_out = r_out.flatten(1)
        feat_res = r_out
        
    return feat_uni.cpu().numpy(), feat_ctr.cpu().numpy(), feat_res.cpu().numpy()

def main():
    print("Loading probe tiles...")
    tiles_df = pd.read_csv("data/processed/classification/tcga_subtypes/manifests/color_probe_samples.csv")
    morph_df = pd.read_parquet(MORPH_STATS_PATH)
    
    print("Loading Generative Models...")
    unet, vae, spatial_encoder, text_encoder, tokenizer, noise_scheduler = load_generative_models()
    
    print("Loading Extractors & MLPs...")
    uni_model, ctr_model, res_model = load_extractors()
    uni_t, ctr_t, res_t = get_transforms()
    
    mlp_uni = joblib.load("artifacts/models/downstream/mlp_uni2h.joblib")
    mlp_ctr = joblib.load("artifacts/models/downstream/mlp_ctranspath.joblib")
    mlp_res = joblib.load("artifacts/models/downstream/mlp_resnet50.joblib")
    
    sweep_dims = [1, 7]
    sweep_vals = [-2.0, -1.0, 0.0, 1.0, 2.0]
    
    results = []
    
    for i, row in tqdm(tiles_df.iterrows(), total=len(tiles_df)):
        stem = Path(row['image_path']).stem
        true_label = row['label']
        
        spatial_path = os.path.join(SPATIAL_DIR, f"{stem}.npz")
        if not os.path.exists(spatial_path):
            continue
            
        spatial_data = np.load(spatial_path)['map']
        base_morph = morph_df.loc[stem].values
        
        # Build batch of morphology vectors
        batch_morphs = [base_morph] # Baseline
        batch_maps = [spatial_data]
        
        for d in sweep_dims:
            for v in sweep_vals:
                mod = base_morph.copy()
                mod[d] = v
                batch_morphs.append(mod)
                batch_maps.append(spatial_data)
                
        # Generate images in one big batch (size 16)
        seed = 42 + i
        images_tensor = generate_batch(unet, vae, spatial_encoder, text_encoder, tokenizer, noise_scheduler, batch_maps, batch_morphs, grid_seed=seed)
        
        # Extract features
        f_uni, f_ctr, f_res = extract_features(images_tensor, uni_model, ctr_model, res_model, uni_t, ctr_t, res_t)
        
        # Predict
        p_uni = mlp_uni.predict(f_uni)
        p_ctr = mlp_ctr.predict(f_ctr)
        p_res = mlp_res.predict(f_res)
        
        # Baseline predictions (index 0)
        base_p_uni = p_uni[0]
        base_p_ctr = p_ctr[0]
        base_p_res = p_res[0]
        
        flipped_uni = False
        flipped_ctr = False
        flipped_res = False
        
        # Check sweeps
        for idx in range(1, len(p_uni)):
            if p_uni[idx] != base_p_uni: flipped_uni = True
            if p_ctr[idx] != base_p_ctr: flipped_ctr = True
            if p_res[idx] != base_p_res: flipped_res = True
            
        results.append({
            "tile": stem,
            "true_label": true_label,
            "base_pred_uni": base_p_uni,
            "base_pred_ctr": base_p_ctr,
            "base_pred_res": base_p_res,
            "flipped_uni": flipped_uni,
            "flipped_ctr": flipped_ctr,
            "flipped_res": flipped_res
        })
        
    res_df = pd.DataFrame(results)
    output_path = Path("artifacts/runs/morphology_invariance/metrics/morph_invariance_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(output_path, index=False)
    
    # Calculate invariance
    inv_uni = 100 * (1 - res_df['flipped_uni'].mean())
    inv_ctr = 100 * (1 - res_df['flipped_ctr'].mean())
    inv_res = 100 * (1 - res_df['flipped_res'].mean())
    
    print("\n=============================")
    print("MORPH INVARIANCE RESULTS")
    print(f"Total Tiles Tested: {len(res_df)}")
    print(f"UNI-2h Morph Invariance: {inv_uni:.1f}%")
    print(f"CTransPath Morph Invariance: {inv_ctr:.1f}%")
    print(f"ResNet50 Morph Invariance: {inv_res:.1f}%")
    print("=============================")

if __name__ == "__main__":
    main()
