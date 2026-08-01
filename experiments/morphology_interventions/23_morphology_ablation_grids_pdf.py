import os
import torch
import numpy as np
import pandas as pd
import random
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import joblib

from diffusers import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import timm
from torchvision import transforms

import sys
sys.path.append(r"src")
from cpathogen.generation.phase2 import SpatialCondEncoder, inject_film_into_unet

# ----------------- Configuration -----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_DTYPE = torch.float16

SPATIAL_DIR = r"data/processed/conditions/spatial_maps"
MORPH_STATS_PATH = r"data/processed/conditions/morphology/standardized.parquet"

OUTPUT_DIR = r"artifacts/runs/morphology_ablation_pdf"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = r"data/raw/tcga_brca/molecular_subtypes.csv"
subtype_df = pd.read_csv(CSV_PATH)
subtype_map = {}
for idx, row in subtype_df.iterrows():
    st = row['molecular_subtype']
    if st in ['BRCA.Luminal A', 'BRCA.Luminal B']:
        st = 'BRCA.Luminal'
    subtype_map[row['sampleID']] = st

MLP_PATH = r"artifacts/models/downstream/classifier.joblib"
SCALER_PATH = r"artifacts/models/downstream/scaler.joblib"

# ----------------- Setup -----------------
print("Loading Morph Data...")
morph_df = pd.read_parquet(MORPH_STATS_PATH)
VARIABLES = list(morph_df.columns)
SWEEP_VALUES = [-2.0, -1.0, 0.0, 1.0, 2.0]

print("Loading PathOGen Models...")
CKPT_DIR = r"artifacts/runs/legacy_phase2_fid58/checkpoints/checkpoint-30000"
BASE_MODEL = "Manojb/stable-diffusion-2-1-base"

from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler

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

print("Loading UNI2-h...")
timm_kwargs = {
    'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
    'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667*2,
    'num_classes': 0, 'no_embed_class': True,
    'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU, 
    'reg_tokens': 8, 'dynamic_img_size': True
}
uni2h = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
uni2h.to(DEVICE)
uni2h.eval()
uni_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

print("Loading MLP Classifier...")
mlp_classifier = joblib.load(MLP_PATH)
scaler = joblib.load(SCALER_PATH)

CLASSES = ["Basal-like", "HER2-enriched", "Luminal", "Normal-like"]

def predict_subtype(image):
    tensor = uni_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad(), torch.cuda.amp.autocast():
        feat = uni2h(tensor)
    feat_np = feat.cpu().numpy()
    feat_scaled = scaler.transform(feat_np)
    probs = mlp_classifier.predict_proba(feat_scaled)[0]
    return probs

def overlay_prediction_bar(image, probs):
    # Add a bar at the bottom with predicted subtype
    bar_height = 30
    new_img = Image.new("RGB", (image.width, image.height + bar_height), "black")
    new_img.paste(image, (0, 0))
    draw = ImageDraw.Draw(new_img)
    
    pred_idx = np.argmax(probs)
    pred_class = CLASSES[pred_idx]
    conf = probs[pred_idx] * 100
    
    color = "white"
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
        
    text = f"{pred_class} ({conf:.1f}%)"
    draw.text((10, image.height + 5), text, fill=color, font=font)
    return new_img

@torch.no_grad()
def generate_fixed_noise_concat_conditioned(unet, vae, spatial_encoder, text_encoder, tokenizer,
                                 noise_scheduler, spatial_maps, morph_vectors, device, weight_dtype,
                                 num_inference_steps=20, grid_seed=42):
    from diffusers import DDIMScheduler
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
    
    unet.eval()
    spatial_encoder.eval()
    
    text_inputs = tokenizer(["he"], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
    text_embeds = text_encoder(text_inputs.input_ids.to(device), return_dict=False)[0]
    
    generated = []
    batch_size = len(spatial_maps)
    
    generator = torch.Generator(device=device).manual_seed(grid_seed)
    base_latent = torch.randn(1, 4, 64, 64, generator=generator, device=device, dtype=weight_dtype)
    
    spatial_tensor = torch.stack([
        torch.from_numpy(sm.astype(np.float32) / 255.0).permute(2, 0, 1)
        for sm in spatial_maps
    ]).to(device, dtype=weight_dtype)
    
    morph_batch = torch.stack([
        mv if isinstance(mv, torch.Tensor) else torch.tensor(mv, dtype=torch.float32)
        for mv in morph_vectors
    ]).to(device, dtype=weight_dtype)

    spatial_features = spatial_encoder(spatial_tensor)
    batch_text_embeds = text_embeds.expand(batch_size, -1, -1)
    
    latents = base_latent.expand(batch_size, -1, -1, -1).clone()
    latents = latents * scheduler.init_noise_sigma

    for module in unet.modules():
        if hasattr(module, "film_mlp"):
            module.current_morph16 = morph_batch
    
    for t in scheduler.timesteps:
        latent_model_input = scheduler.scale_model_input(latents, t)
        unet_input = torch.cat([latent_model_input, spatial_features], dim=1)
        
        with torch.autocast("cuda", dtype=weight_dtype):
            noise_pred = unet(unet_input, t, encoder_hidden_states=batch_text_embeds, return_dict=False)[0]
        
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    
    latents_for_decode = latents / vae.config.scaling_factor
    images = vae.decode(latents_for_decode, return_dict=False)[0]
    
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    for img_np in images:
        pil_img = Image.fromarray((img_np * 255).astype(np.uint8))
        generated.append(pil_img)
            
    for module in unet.modules():
        if hasattr(module, "film_mlp"):
            module.current_morph16 = None
            
    return generated

def main():
    all_spatial_files = list(Path(SPATIAL_DIR).glob("*.npz"))
    valid_files = [f for f in all_spatial_files if f.stem in morph_df.index]
    
    # Filter to only known subtypes
    known_files = []
    for f in valid_files:
        patient_id = "-".join(f.stem.split("_")[0].split("-")[:3])
        if patient_id in subtype_map and subtype_map[patient_id] != "Unknown":
            known_files.append(f)
            
    # Make a mixed bag of ~10 tissues, ensuring the important ones are included
    important_stems = [
        "TCGA-A2-A0CW_x65536_y47104_TL",
        "TCGA-A8-A083_x14336_y30720_BR"
    ]
    
    target_files = []
    # Add important files first
    for f in known_files:
        if f.stem in important_stems:
            target_files.append(f)
            
    # Sample remaining to get a total of 10
    remaining_files = [f for f in known_files if f.stem not in important_stems]
    random.seed(123)
    target_files += random.sample(remaining_files, min(8, len(remaining_files)))
    
    for fpath in tqdm(target_files, desc="Generating Ablation Grids"):
        stem = fpath.stem
        spatial_data = np.load(fpath)['map']
        base_morph = morph_df.loc[stem].values
        
        # Grid settings
        cell_size = 512
        bar_height = 30
        margin = 50
        header_height = 50
        
        grid_width = margin + len(SWEEP_VALUES) * cell_size
        grid_height = header_height + len(VARIABLES) * (cell_size + bar_height)
        
        grid_img = Image.new("RGB", (grid_width, grid_height), "white")
        draw = ImageDraw.Draw(grid_img)
        
        try:
            title_font = ImageFont.truetype("arial.ttf", 24)
            label_font = ImageFont.truetype("arial.ttf", 20)
        except:
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()
            
        # Draw Column Headers
        for j, val in enumerate(SWEEP_VALUES):
            text = f"{val} std"
            x = margin + j * cell_size + (cell_size // 2) - 30
            y = 10
            draw.text((x, y), text, fill="black", font=title_font)
            
        if stem == "TCGA-A1-A0SE_x76800_y52224_BL":
            grid_seed = 9999
        else:
            grid_seed = random.randint(0, 100000)
        
        # We process one variable at a time (one row)
        for i, var_name in enumerate(VARIABLES):
            y_offset = header_height + i * (cell_size + bar_height)
            
            # Draw Row Label (Variable Name)
            # Rotate text: create small image, draw, rotate, paste
            txt_img = Image.new("RGBA", (200, margin), (255, 255, 255, 0))
            txt_draw = ImageDraw.Draw(txt_img)
            txt_draw.text((10, 10), var_name, fill="black", font=label_font)
            txt_img = txt_img.rotate(90, expand=True)
            grid_img.paste(txt_img, (5, y_offset + (cell_size // 2) - 100), txt_img)
            
            # Prepare batch for this row (5 values)
            batch_morphs = []
            for val in SWEEP_VALUES:
                mod_morph = base_morph.copy()
                mod_morph[i] = val
                batch_morphs.append(torch.tensor(mod_morph, dtype=torch.float32))
                
            batch_maps = [spatial_data] * len(SWEEP_VALUES)
            
            # Generate row
            gen_images = generate_fixed_noise_concat_conditioned(
                unet, vae, spatial_encoder, text_encoder, tokenizer,
                noise_scheduler, batch_maps, batch_morphs, DEVICE, WEIGHT_DTYPE,
                num_inference_steps=20, grid_seed=grid_seed
            )
            
            # Overlay predictions and paste into grid
            for j, img in enumerate(gen_images):
                probs = predict_subtype(img)
                img_with_bar = overlay_prediction_bar(img, probs)
                
                x_offset = margin + j * cell_size
                grid_img.paste(img_with_bar, (x_offset, y_offset))
                
        patient_id = "-".join(stem.split("_")[0].split("-")[:3])
        gt_class = subtype_map.get(patient_id, "Unknown").replace("BRCA.", "")
        
        out_path = os.path.join(OUTPUT_DIR, f"{stem}_GT-{gt_class}_ablation.pdf")
        grid_img.save(out_path, "PDF", resolution=100.0)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
