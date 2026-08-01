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

def train_mlp_classifier(uni_model, uni_transform, device):
    import glob
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
    random.seed(42)
    selected_tiles = {p: [random.choice(patient_to_images[p])] for p in valid_patients}
    train_patients, test_patients = train_test_split(valid_patients, test_size=0.2, random_state=42)

    all_labels = [subtype_map[p] for p in valid_patients]
    le = LabelEncoder()
    le.fit(all_labels)

    def extract_features(patient_list):
        features, labels = [], []
        with torch.no_grad():
            for idx, p in enumerate(patient_list):
                label_encoded = le.transform([subtype_map[p]])[0]
                for img_path in selected_tiles[p]:
                    img = Image.open(img_path).convert("RGB")
                    tensor = uni_transform(img).unsqueeze(0).to(device)
                    out = uni_model(tensor)
                    features.append(out[0].cpu().numpy())
                    labels.append(label_encoded)
        return np.array(features), np.array(labels)

    X_train, y_train = extract_features(train_patients)
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
    batch_size = 5
    
    # -------------------------------------------------------------
    # FIXED NOISE MATRICES FOR THE ENTIRE BATCH / GRID
    # Generate ONE noise matrix, and clone it for the whole batch
    # -------------------------------------------------------------
    generator = torch.Generator(device=device).manual_seed(grid_seed)
    base_latent = torch.randn(1, 4, 64, 64, generator=generator, device=device, dtype=weight_dtype)
    
    for i in range(0, len(spatial_maps), batch_size):
        current_batch = spatial_maps[i:i+batch_size]
        bs = len(current_batch)
        
        spatial_tensor = torch.stack([
            torch.from_numpy(sm.astype(np.float32) / 255.0).permute(2, 0, 1)
            for sm in current_batch
        ]).to(device, dtype=weight_dtype)
        
        morph_batch = torch.stack([
            mv if isinstance(mv, torch.Tensor) else torch.tensor(mv, dtype=torch.float32)
            for mv in morph_vectors[i:i+batch_size]
        ]).to(device, dtype=weight_dtype)

        spatial_features = spatial_encoder(spatial_tensor)
        batch_text_embeds = text_embeds.expand(bs, -1, -1)
        
        # Clone exactly
        latents = base_latent.expand(bs, -1, -1, -1).clone()
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

def add_prediction_label(img, class_name, probability):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except:
        font = ImageFont.load_default()
    
    draw.rectangle([0, 0, img.width, 35], fill=(0, 0, 0, 200))
    display_name = class_name.replace("BRCA.", "")
    draw.text((10, 5), f"{display_name}: {probability*100:.1f}%", fill="white", font=font)
    return img

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_dtype = torch.float16
    
    print("Loading UNI2-h model...")
    uni_model, uni_transform = load_uni2h_model(device)
    
    print("Training Molecular MLP Classifier...")
    clf, label_encoder = train_mlp_classifier(uni_model, uni_transform, device)
    classes = label_encoder.classes_
    
    print("Loading PathOGen generative models...")
    unet, vae, spatial_encoder, text_encoder, tokenizer, noise_scheduler = load_pathogen_models(device, weight_dtype)
    
    DATA_DIR = Path(r"data/processed/generator")
    spatial_dir = Path("data/processed/generator/spatial_maps")
    all_spatial_files = list(spatial_dir.glob("*.npz"))
    
    morph_path = Path("data/processed/generator/morphology_features/morphology_standardized.parquet")
    morph_df = pd.read_parquet(morph_path)
    
    valid_files = [f for f in all_spatial_files if f.stem in morph_df.index]
    
    OUTPUT_DIR = Path(r"artifacts/runs/large_scale_molecular_probe")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grids_dir = OUTPUT_DIR / "grids"
    grids_dir.mkdir(exist_ok=True)
    
    N_GRIDS = 100
    N_SWAPS = 20 # 5x4 grid
    
    random.seed(1234) # Seed for selecting target tiles
    target_files = random.sample(valid_files, min(N_GRIDS, len(valid_files)))
    all_stems = list(morph_df.index)
    
    results = []
    
    print(f"Starting 100-Grid Probe on {len(target_files)} target tiles...")
    
    for idx, spatial_path in enumerate(tqdm(target_files, desc="Tiles")):
        stem = spatial_path.stem
        try:
            # 1. Exact same spatial map for all 20 images in the grid
            original_map = np.load(spatial_path)["map"]
            batch_maps = [original_map] * N_SWAPS
            
            # 2. Random morphological embeddings for each of the 20 images
            swap_stems = random.sample(all_stems, N_SWAPS)
            swap_morphs = [torch.tensor(morph_df.loc[s].values, dtype=torch.float32) for s in swap_stems]
            
            # 3. Unique random noise matrix seed FOR THIS SPECIFIC GRID
            grid_seed = random.randint(0, 1000000)
            
            # 4. Generate the 20 images. The custom function will force `grid_seed`
            # to be identical across all 20 instances in the batch/grid!
            gen_images = generate_fixed_noise_concat_conditioned(
                unet, vae, spatial_encoder, text_encoder, tokenizer,
                noise_scheduler, batch_maps, swap_morphs, device, weight_dtype,
                num_inference_steps=20, grid_seed=grid_seed
            )
            
            grid_images = []
            
            # 5. Extract UNI2-h embeddings and MLP predictions
            for j, gen_img in enumerate(gen_images):
                swap_stem = swap_stems[j]
                
                tensor = uni_transform(gen_img.convert('RGB')).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = uni_model(tensor).cpu().numpy()
                    
                probs = clf.predict_proba(embedding)[0]
                pred_idx = np.argmax(probs)
                pred_class = classes[pred_idx]
                prob_val = probs[pred_idx]
                
                results.append({
                    "Grid_Index": idx,
                    "Target_Spatial_Stem": stem,
                    "Grid_Seed": grid_seed,
                    "Swap_Morphology_Stem": swap_stem,
                    "Predicted_Class": pred_class,
                    "Predicted_Prob": prob_val,
                    "Prob_Basal": probs[0],
                    "Prob_HER2": probs[1],
                    "Prob_Luminal": probs[2],
                    "Prob_Normal": probs[3]
                })
                
                labeled_img = add_prediction_label(gen_img.copy(), pred_class, prob_val)
                grid_images.append(labeled_img)
        
            # 6. Assemble 5 columns x 4 rows grid (20 images)
            grid = Image.new("RGB", (512 * 5, 512 * 4))
            for i, img in enumerate(grid_images):
                row = i // 5
                col = i % 5
                grid.paste(img, (512 * col, 512 * row))
                
            grid.save(grids_dir / f"{stem}_morph_probe.png")
            
        except Exception as e:
            print(f"Error processing {stem}: {e}")
            
    df_results = pd.DataFrame(results)
    output_csv = OUTPUT_DIR / "large_scale_molecular_probe_results.csv"
    df_results.to_csv(output_csv, index=False)
    print(f"\nSaved aggregated results to {output_csv}")

if __name__ == "__main__":
    main()
