import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
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
from cpathogen.generation.inference import generate_concat_conditioned

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
    print("Training Molecular MLP Classifier...")
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

def add_prediction_label(img, class_name, probability, row_idx, col_idx):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
        font_small = ImageFont.truetype("arial.ttf", 16)
    except:
        font = font_small = ImageFont.load_default()
    
    # Draw a black rectangle at the top for visibility
    draw.rectangle([0, 0, img.width, 50], fill=(0, 0, 0, 200))
    
    # Format class name to be shorter if needed
    display_name = class_name.replace("BRCA.", "")
    draw.text((10, 5), f"{display_name}: {probability*100:.1f}%", fill="white", font=font)
    
    # Draw Row/Col info for clarity
    draw.text((10, 32), f"Morph {row_idx+1} | Spat {col_idx+1}", fill="lightgray", font=font_small)
    
    return img

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_dtype = torch.float16
    
    print("Loading UNI2-h model...")
    uni_model, uni_transform = load_uni2h_model(device)
    
    clf, label_encoder = train_mlp_classifier(uni_model, uni_transform, device)
    classes = label_encoder.classes_
    
    print("Loading PathOGen generative models...")
    unet, vae, spatial_encoder, text_encoder, tokenizer, noise_scheduler = load_pathogen_models(device, weight_dtype)
    
    DATA_DIR = Path(r"data/processed/conditions")
    spatial_dir = Path("data/processed/conditions/spatial_maps")
    all_spatial_files = list(spatial_dir.glob("*.npz"))
    
    morph_path = Path("data/processed/conditions/morphology/standardized.parquet")
    morph_df = pd.read_parquet(morph_path)
    
    valid_files = [f for f in all_spatial_files if f.stem in morph_df.index]
    
    OUTPUT_DIR = Path(r"artifacts/runs/molecular_mlp_probe")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    N_GRIDS = 5
    COLS = 5 # 5 different spatial maps / noises per grid
    ROWS = 4 # 4 different morphological vectors per grid
    
    random.seed(999)
    
    print(f"Generating {N_GRIDS} Grids...")
    for grid_idx in range(N_GRIDS):
        print(f"\nProcessing Grid {grid_idx+1}/{N_GRIDS}")
        
        # Select 5 random files for Columns (Spatial + Seed)
        col_files = random.sample(valid_files, COLS)
        col_maps = [np.load(f)["map"] for f in col_files]
        col_seeds = [random.randint(0, 100000) for _ in range(COLS)]
        
        # Select 4 random files for Rows (Morphology)
        row_stems = random.sample(list(morph_df.index), ROWS)
        row_morphs = [torch.tensor(morph_df.loc[s].values, dtype=torch.float32) for s in row_stems]
        
        grid_images = []
        
        # Generate image for each cell in the grid
        # We process row by row, but generate cell by cell because seeds depend on column
        for r in range(ROWS):
            for c in range(COLS):
                # We need to run generation individually because seeds differ per column,
                # or we can pass a list of generators. `generate_concat_conditioned` in inference
                # takes a single seed integer. So we will just loop.
                gen_image = generate_concat_conditioned(
                    unet, vae, spatial_encoder, text_encoder, tokenizer,
                    noise_scheduler, [col_maps[c]], [row_morphs[r]], device, weight_dtype,
                    num_inference_steps=20, seed=col_seeds[c]
                )[0]
                
                # Inference
                tensor = uni_transform(gen_image.convert('RGB')).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = uni_model(tensor).cpu().numpy()
                
                probs = clf.predict_proba(embedding)[0]
                pred_idx = np.argmax(probs)
                pred_class = classes[pred_idx]
                prob_val = probs[pred_idx]
                
                labeled_img = add_prediction_label(gen_image.copy(), pred_class, prob_val, r, c)
                grid_images.append(labeled_img)
                
        # Assemble 5 columns x 4 rows grid (20 images)
        grid = Image.new("RGB", (512 * COLS, 512 * ROWS))
        for i, img in enumerate(grid_images):
            row = i // COLS
            col = i % COLS
            grid.paste(img, (512 * col, 512 * row))
            
        grid.save(OUTPUT_DIR / f"molecular_probe_grid_{grid_idx+1}.png")
        print(f"Saved Grid {grid_idx+1}")

if __name__ == "__main__":
    main()
