import os
import glob
import pandas as pd
import numpy as np
import torch
import random
from PIL import Image
from tqdm import tqdm

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading UNI2-h model...")
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

    print("Loading molecular subtypes map...")
    CSV_PATH = r"data/raw/tcga_brca/molecular_subtypes.csv"
    df = pd.read_csv(CSV_PATH)
    subtype_map = {}
    for idx, row in df.iterrows():
        subtype = row['molecular_subtype']
        # Combine Luminal A and B
        if subtype in ['BRCA.Luminal A', 'BRCA.Luminal B']:
            subtype = 'BRCA.Luminal'
        subtype_map[row['sampleID']] = subtype

    print("Scanning images directory...")
    IMAGES_DIR = r"data/interim/tiles/tcga_brca"
    all_images = glob.glob(os.path.join(IMAGES_DIR, "*.png"))

    patient_to_images = {}
    for img_path in all_images:
        basename = os.path.basename(img_path)
        if not basename.startswith("TCGA-"):
            continue
        parts = basename.split("_")[0].split("-")
        if len(parts) >= 3:
            patient_id = "-".join(parts[:3])
            if patient_id in subtype_map:
                if patient_id not in patient_to_images:
                    patient_to_images[patient_id] = []
                patient_to_images[patient_id].append(img_path)

    valid_patients = [p for p, imgs in patient_to_images.items() if len(imgs) >= 1]
    print(f"Found {len(valid_patients)} valid patients.")

    # Patient-level Train/Test split to prevent data leakage
    train_patients, test_patients = train_test_split(valid_patients, test_size=0.2, random_state=42)

    train_classes = [subtype_map[p] for p in train_patients]
    test_classes = [subtype_map[p] for p in test_patients]
    
    unique_classes = sorted(list(set(train_classes)))
    print(f"Classes: {unique_classes}")

    # --- BALANCING LOGIC FOR TRAINING SET ---
    print("\nBalancing Training Data...")
    from collections import Counter
    train_class_counts = Counter(train_classes)
    
    # Target number of samples per class (set to majority class count)
    target_count = max(train_class_counts.values())
    print(f"Target samples per class: {target_count}")
    
    train_images = []
    train_labels_str = []
    
    for c in unique_classes:
        # Get all training patients in this class
        patients_in_c = [p for p in train_patients if subtype_map[p] == c]
        
        # Pool ALL available tiles for these patients
        all_tiles_for_c = []
        for p in patients_in_c:
            all_tiles_for_c.extend(patient_to_images[p])
            
        # Sample 'target_count' tiles
        if len(all_tiles_for_c) >= target_count:
            # We have enough unique tiles, sample without replacement
            sampled_tiles = random.sample(all_tiles_for_c, target_count)
            print(f"  {c}: Pooled {len(all_tiles_for_c)} tiles -> Sampled {target_count} unique tiles.")
        else:
            # We don't have enough unique tiles, sample with replacement (oversampling)
            sampled_tiles = random.choices(all_tiles_for_c, k=target_count)
            unique_sampled = len(set(sampled_tiles))
            print(f"  {c}: Pooled {len(all_tiles_for_c)} tiles -> Oversampled to {target_count} tiles (used {unique_sampled} unique tiles).")
            
        train_images.extend(sampled_tiles)
        train_labels_str.extend([c] * target_count)

    # --- TEST SET PREPARATION ---
    # To keep evaluation strictly comparable to the previous 1-tile baseline, 
    # we evaluate on EXACTLY 1 random tile per test patient.
    test_images = []
    test_labels_str = []
    for p in test_patients:
        test_images.append(random.choice(patient_to_images[p]))
        test_labels_str.append(subtype_map[p])

    # Encode labels
    le = LabelEncoder()
    le.fit(unique_classes)
    y_train = le.transform(train_labels_str)
    y_test = le.transform(test_labels_str)

    # --- FEATURE EXTRACTION ---
    def extract_features(image_paths, desc):
        features = []
        with torch.no_grad():
            for img_path in tqdm(image_paths, desc=desc):
                img = Image.open(img_path).convert("RGB")
                tensor = uni_transform(img).unsqueeze(0).to(device)
                out = uni_model(tensor)
                features.append(out[0].cpu().numpy())
        return np.array(features)

    print("\nExtracting features for balanced training set...")
    X_train = extract_features(train_images, "Train Ext")
    
    print("\nExtracting features for 1-tile test set...")
    X_test = extract_features(test_images, "Test Ext")

    # --- MODEL TRAINING & EVALUATION ---
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(512, 128), max_iter=500, random_state=42)
    }

    # Binarize labels for per-class AUC
    lb = LabelEncoder()
    lb.fit(unique_classes)
    y_test_bin = np.zeros((len(y_test), len(unique_classes)))
    for i, label in enumerate(y_test):
        y_test_bin[i, label] = 1

    for name, model in models.items():
        print(f"\n==================================================")
        print(f"Training {name} on balanced UNI2-h features...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"[{name}] Accuracy: {acc*100:.2f}%")
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            try:
                overall_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
                print(f"[{name}] Overall AUC (OvR): {overall_auc:.4f}")
                
                print(f"--- Per-Class AUC ({name}) ---")
                for i, class_name in enumerate(unique_classes):
                    try:
                        auc = roc_auc_score(y_test_bin[:, i], y_prob[:, i])
                        print(f"  {class_name}: {auc:.4f}")
                    except ValueError:
                        print(f"  {class_name}: AUC undefined (no positive samples in test)")
            except ValueError as e:
                print(f"[{name}] AUC Error: {e}")
        else:
            print(f"[{name}] does not support predict_proba for AUC.")

if __name__ == "__main__":
    main()
