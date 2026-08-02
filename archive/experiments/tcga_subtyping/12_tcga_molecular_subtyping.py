import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

# Configure Paths
CSV_PATH = r"data/raw/tcga_brca/molecular_subtypes.csv"
IMAGES_DIR = r"data/interim/tiles/tcga_brca"

def main():
    print("Loading molecular subtypes map...")
    df = pd.read_csv(CSV_PATH)
    subtype_map = dict(zip(df['sampleID'], df['molecular_subtype']))

    print("Scanning images directory...")
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

    # Filter patients with < 4 tiles (still enforce a minimum to have reliable patient aggregation)
    valid_patients = [p for p, imgs in patient_to_images.items() if len(imgs) >= 4]
    print(f"Found {len(valid_patients)} valid patients.")
    
    # We will use ALL available tiles for each valid patient to maximize data
    selected_tiles = {p: patient_to_images[p] for p in valid_patients}
    total_tiles = sum(len(imgs) for imgs in selected_tiles.values())
    print(f"Total tiles across all valid patients: {total_tiles}")

    # Split patients into Train/Test (80/20)
    train_patients, test_patients = train_test_split(valid_patients, test_size=0.2, random_state=42)
    print(f"Train patients: {len(train_patients)}, Test patients: {len(test_patients)}")

    # Encode labels for XGBoost and general usage
    all_labels = [subtype_map[p] for p in valid_patients]
    le = LabelEncoder()
    le.fit(all_labels)
    classes = le.classes_
    print(f"Classes: {classes}")

    print("Loading UNI2-h model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timm_kwargs = {
        'img_size': 224, 
        'patch_size': 14, 
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5, 
        'embed_dim': 1536,
        'mlp_ratio': 2.66667*2,
        'num_classes': 0, 
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 
        'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 
        'dynamic_img_size': True
    }
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    def extract_features(patient_list):
        features = []
        labels = []
        patient_ids = []
        
        with torch.no_grad():
            for idx, p in enumerate(patient_list):
                if idx % 10 == 0:
                    print(f"  Processing patient {idx}/{len(patient_list)}...")
                
                label_encoded = le.transform([subtype_map[p]])[0]
                
                # Process in batches of 16 to avoid OOM
                p_tiles = selected_tiles[p]
                BATCH_SIZE = 16
                for i in range(0, len(p_tiles), BATCH_SIZE):
                    batch_files = p_tiles[i:i+BATCH_SIZE]
                    batch_tensors = []
                    for img_path in batch_files:
                        img = Image.open(img_path).convert("RGB")
                        tensor = transform(img)
                        batch_tensors.append(tensor)
                    
                    batch = torch.stack(batch_tensors).to(device)
                    out = model(batch)
                    
                    for feat in out.cpu().numpy():
                        features.append(feat)
                        labels.append(label_encoded)
                        patient_ids.append(p)
                    
        return np.array(features), np.array(labels), np.array(patient_ids)

    print("Extracting features for training set...")
    X_train, y_train, p_train = extract_features(train_patients)
    print(f"Training set tiles: {len(X_train)}")
    
    print("Extracting features for test set...")
    X_test, y_test, p_test = extract_features(test_patients)
    print(f"Testing set tiles: {len(X_test)}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
        "MLP (Neural Network)": MLPClassifier(hidden_layer_sizes=(512, 128), max_iter=500, random_state=42)
    }

    lb = LabelBinarizer()
    lb.fit(y_train)

    for model_name, clf in models.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}...")
        clf.fit(X_train, y_train)

        print(f"Evaluating {model_name}...")
        patient_true = []
        patient_pred_probs = []
        
        for p in test_patients:
            idx = np.where(p_test == p)[0]
            if len(idx) == 0:
                continue
            
            # shape: (num_tiles, num_classes)
            probs = clf.predict_proba(X_test[idx])
            avg_prob = np.mean(probs, axis=0)
            
            patient_true.append(le.transform([subtype_map[p]])[0])
            patient_pred_probs.append(avg_prob)

        patient_true = np.array(patient_true)
        patient_pred_probs = np.array(patient_pred_probs)
        patient_preds = np.argmax(patient_pred_probs, axis=1)

        accuracy = accuracy_score(patient_true, patient_preds)
        print(f"[{model_name}] Patient-Level Accuracy: {accuracy * 100:.2f}%")
        
        try:
            y_true_bin = lb.transform(patient_true)
            # Calculate overall AUC
            overall_auc = roc_auc_score(y_true_bin, patient_pred_probs, multi_class="ovr")
            print(f"[{model_name}] Overall Patient-Level AUC (OvR): {overall_auc:.4f}\n")
            
            # Calculate per-class AUC
            print(f"--- Per-Class AUC ({model_name}) ---")
            for i, class_name in enumerate(classes):
                # Ensure the class is present in the test set to avoid errors
                if len(np.unique(y_true_bin[:, i])) > 1:
                    class_auc = roc_auc_score(y_true_bin[:, i], patient_pred_probs[:, i])
                    print(f"  {class_name}: {class_auc:.4f}")
                else:
                    print(f"  {class_name}: N/A (not enough samples in test set)")
        except Exception as e:
            print(f"Could not compute AUC for {model_name}: {e}")

if __name__ == "__main__":
    main()
