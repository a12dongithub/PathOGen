import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import random
from tqdm import tqdm
from scipy.ndimage import label
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

import sys
sys.path.append(r"src")
from cpathogen.generation.phase2 import SpatialCondEncoder, inject_film_into_unet

def load_pathogen_models(device, weight_dtype):
    CKPT_DIR = r"artifacts/runs/legacy_phase2_fid58/checkpoints/checkpoint-30000"
    BASE_MODEL = "Manojb/stable-diffusion-2-1-base"
    
    tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL, subfolder="text_encoder", torch_dtype=weight_dtype).to(device)
    vae = AutoencoderKL.from_pretrained(BASE_MODEL, subfolder="vae", torch_dtype=weight_dtype).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")
    
    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet", torch_dtype=weight_dtype)
    old_conv_in = unet.conv_in
    new_conv_in = torch.nn.Conv2d(8, old_conv_in.out_channels, kernel_size=old_conv_in.kernel_size,
                                  stride=old_conv_in.stride, padding=old_conv_in.padding).to(unet.device, dtype=weight_dtype)
    with torch.no_grad():
        new_conv_in.weight[:, :4] = old_conv_in.weight
        new_conv_in.weight[:, 4:] = 0.0
        new_conv_in.bias.copy_(old_conv_in.bias)
    unet.conv_in = new_conv_in
    unet.config['in_channels'] = 8

    film_mlps = inject_film_into_unet(unet, film_dim=16)
    unet.load_state_dict(UNet2DConditionModel.from_pretrained(CKPT_DIR, subfolder="unet").state_dict(), strict=False)
    unet.to(device)
    
    spatial_encoder = SpatialCondEncoder().to(device, dtype=weight_dtype)
    spatial_encoder.load_state_dict(torch.load(os.path.join(CKPT_DIR, "spatial_encoder.pt"), map_location=device))
    film_mlps.load_state_dict(torch.load(os.path.join(CKPT_DIR, "film_mlps.pt"), map_location=device))
    film_mlps.to(device, dtype=weight_dtype)
    
    return unet, vae, spatial_encoder, text_encoder, tokenizer, noise_scheduler

def load_uni2h_model(device):
    timm_kwargs = {
        'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
        'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667*2,
        'num_classes': 0, 'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 'dynamic_img_size': True
    }
    uni_model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    uni_transform = create_transform(**resolve_data_config(uni_model.pretrained_cfg, model=uni_model))
    uni_model.eval().to(device)
    return uni_model, uni_transform

def train_balanced_mlp_classifier(uni_model, uni_transform, device):
    import glob
    from collections import Counter
    
    CSV_PATH = r"data/raw/tcga_brca/molecular_subtypes.csv"
    IMAGES_DIR = r"data/interim/tiles/tcga_brca"

    df = pd.read_csv(CSV_PATH)
    subtype_map = {}
    for idx, row in df.iterrows():
        subtype = row['molecular_subtype']
        if subtype in ['BRCA.Luminal A', 'BRCA.Luminal B']:
            subtype = 'BRCA.Luminal'
        subtype_map[row['sampleID']] = subtype

    all_images = glob.glob(os.path.join(IMAGES_DIR, "*.png"))
    patient_to_images = {}
    for img_path in all_images:
        basename = os.path.basename(img_path)
        if not basename.startswith("TCGA-"): continue
        parts = basename.split("_")[0].split("-")
        if len(parts) >= 3:
            patient_id = "-".join(parts[:3])
            if patient_id in subtype_map:
                if patient_id not in patient_to_images:
                    patient_to_images[patient_id] = []
                patient_to_images[patient_id].append(img_path)

    valid_patients = [p for p, imgs in patient_to_images.items() if len(imgs) >= 1]
    
    train_patients, _ = train_test_split(valid_patients, test_size=0.2, random_state=42)

    train_classes = [subtype_map[p] for p in train_patients]
    unique_classes = sorted(list(set(train_classes)))
    
    train_class_counts = Counter(train_classes)
    target_count = max(train_class_counts.values())
    
    train_images = []
    train_labels_str = []
    
    random.seed(42)
    for c in unique_classes:
        patients_in_c = [p for p in train_patients if subtype_map[p] == c]
        all_tiles_for_c = []
        for p in patients_in_c:
            all_tiles_for_c.extend(patient_to_images[p])
            
        if len(all_tiles_for_c) >= target_count:
            sampled_tiles = random.sample(all_tiles_for_c, target_count)
        else:
            sampled_tiles = random.choices(all_tiles_for_c, k=target_count)
            
        train_images.extend(sampled_tiles)
        train_labels_str.extend([c] * target_count)

    le = LabelEncoder()
    le.fit(unique_classes)
    y_train = le.transform(train_labels_str)

    def extract_features(image_paths, desc):
        features = []
        with torch.no_grad():
            for img_path in tqdm(image_paths, desc=desc):
                img = Image.open(img_path).convert("RGB")
                tensor = uni_transform(img).unsqueeze(0).to(device)
                out = uni_model(tensor)
                features.append(out[0].cpu().numpy())
        return np.array(features)

    print("\nExtracting balanced training set features...")
    X_train = extract_features(train_images, "Train Ext")
    
    print("\nTraining Balanced MLP...")
    clf = MLPClassifier(hidden_layer_sizes=(512, 128), max_iter=500, random_state=42)
    clf.fit(X_train, y_train)
    return clf, le

@torch.no_grad()
def generate_fixed_noise_concat_conditioned(unet, vae, spatial_encoder, text_encoder, tokenizer,
                                 noise_scheduler, spatial_maps, morph_vectors, device, weight_dtype,
                                 num_inference_steps=20, grid_seed=42):
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
    unet.train()
    spatial_encoder.train()
    
    return generated

def modify_spatial_map(original_map, fraction_to_convert):
    new_map = np.copy(original_map)
    
    mask_0 = new_map[:, :, 0] > 0
    mask_4 = new_map[:, :, 4] > 0
    
    labeled_0, num_0 = label(mask_0)
    labeled_4, num_4 = label(mask_4)
    
    all_cells = []
    for i in range(1, num_0 + 1):
        all_cells.append((0, i))
    for i in range(1, num_4 + 1):
        all_cells.append((4, i))
        
    num_to_convert = int(len(all_cells) * fraction_to_convert)
    if num_to_convert > 0:
        cells_to_convert = random.sample(all_cells, num_to_convert)
        for ch, lbl in cells_to_convert:
            if ch == 0:
                cell_mask = (labeled_0 == lbl)
            else:
                cell_mask = (labeled_4 == lbl)
                
            new_map[:, :, 1][cell_mask] = new_map[:, :, ch][cell_mask]
            new_map[:, :, ch][cell_mask] = 0
            
    return new_map

def add_prediction_label(img, class_name, probability, subtitle=""):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
        font_sub = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
        font_sub = ImageFont.load_default()
    
    draw.rectangle([0, 0, img.width, 60], fill=(0, 0, 0, 200))
    display_name = class_name.replace("BRCA.", "")
    draw.text((10, 5), f"{display_name}: {probability*100:.1f}%", fill="white", font=font)
    draw.text((10, 35), subtitle, fill=(200, 200, 200), font=font_sub)
    return img

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_dtype = torch.float16
    
    print("Loading UNI2-h model...")
    uni_model, uni_transform = load_uni2h_model(device)
    
    print("Training Balanced Molecular MLP Classifier...")
    clf, label_encoder = train_balanced_mlp_classifier(uni_model, uni_transform, device)
    classes = label_encoder.classes_
    
    print("Loading PathOGen generative models...")
    unet, vae, spatial_encoder, text_encoder, tokenizer, noise_scheduler = load_pathogen_models(device, weight_dtype)
    
    TARGET_STEM = "TCGA-A1-A0SE_x76800_y52224_BL"
    DATA_DIR = Path(r"data/processed/conditions")
    
    spatial_path = Path("data/processed/conditions/spatial_maps") / f"{TARGET_STEM}.npz"
    morph_path = Path("data/processed/conditions/morphology/standardized.parquet")
        
    morph_df = pd.read_parquet(morph_path)
    original_map = np.load(spatial_path)["map"]
    morph_vector = torch.tensor(morph_df.loc[TARGET_STEM].values, dtype=torch.float32)
    
    fractions = [0.0, 0.25, 0.50, 0.75, 1.0]
    batch_maps = []
    
    random.seed(42)
    print("Generating modified spatial maps...")
    for f in fractions:
        mod_map = modify_spatial_map(original_map, f)
        batch_maps.append(mod_map)
        
    batch_morphs = [morph_vector] * len(fractions)
    grid_seed = 9999
    
    print("Running PathOGen inference...")
    gen_images = generate_fixed_noise_concat_conditioned(
        unet, vae, spatial_encoder, text_encoder, tokenizer,
        noise_scheduler, batch_maps, batch_morphs, device, weight_dtype,
        num_inference_steps=20, grid_seed=grid_seed
    )
    
    labeled_images = []
    print("Extracting features and predicting molecular subtypes...")
    for i, gen_img in enumerate(gen_images):
        tensor = uni_transform(gen_img.convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = uni_model(tensor).cpu().numpy()
            
        probs = clf.predict_proba(embedding)[0]
        pred_idx = np.argmax(probs)
        pred_class = classes[pred_idx]
        prob_val = probs[pred_idx]
        
        subtitle = f"{fractions[i]*100:.0f}% Tissue -> Immune"
        labeled = add_prediction_label(gen_img.copy(), pred_class, prob_val, subtitle=subtitle)
        labeled_images.append(labeled)
        
    grid = Image.new("RGB", (512 * len(fractions), 512))
    for i, img in enumerate(labeled_images):
        grid.paste(img, (512 * i, 0))
        
    OUTPUT_DIR = Path(r"artifacts/runs/large_scale_balanced_morph_probe")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"{TARGET_STEM}_cellular_shift_probe.png"
    grid.save(out_file)
    print(f"Saved results to {out_file}")

if __name__ == "__main__":
    main()
