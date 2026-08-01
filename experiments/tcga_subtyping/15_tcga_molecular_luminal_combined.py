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

CSV_PATH = r"data/raw/tcga_brca/molecular_subtypes.csv"
IMAGES_DIR = r"data/interim/tiles/tcga_brca"

def main():
    print("Loading molecular subtypes map...")
    df = pd.read_csv(CSV_PATH)
    
    # Combine Luminal A and Luminal B into Luminal
    subtype_map = {}
    for idx, row in df.iterrows():
        subtype = row['molecular_subtype']
        if subtype in ['BRCA.Luminal A', 'BRCA.Luminal B']:
            subtype = 'BRCA.Luminal'
        subtype_map[row['sampleID']] = subtype

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

    valid_patients = [p for p, imgs in patient_to_images.items() if len(imgs) >= 1]
    print(f"Found {len(valid_patients)} valid patients.")
    
    # Select exactly 1 random tile per patient
    random.seed(42)
    selected_tiles = {p: [random.choice(patient_to_images[p])] for p in valid_patients}
    total_tiles = sum(len(imgs) for imgs in selected_tiles.values())
    print(f"Total tiles (1 per patient): {total_tiles}")

    train_patients, test_patients = train_test_split(valid_patients, test_size=0.2, random_state=42)
    print(f"Train patients: {len(train_patients)}, Test patients: {len(test_patients)}")

    all_labels = [subtype_map[p] for p in valid_patients]
    le = LabelEncoder()
    le.fit(all_labels)
    classes = le.classes_
    print(f"Classes: {classes}")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for encoder_name, load_fn in [
        ("UNI2-h", lambda: _load_uni2h(device)),
        ("ResNet50 (CNN)", lambda: _load_resnet50(device)),
    ]:
        print(f"\n{'#'*60}")
        print(f"# ENCODER: {encoder_name}")
        print(f"{'#'*60}")

        model = load_fn()
        model.eval()

        def extract_features(patient_list):
            features, labels, patient_ids = [], [], []
            with torch.no_grad():
                for idx, p in enumerate(patient_list):
                    if idx % 50 == 0:
                        print(f"  Processing patient {idx}/{len(patient_list)}...")
                    label_encoded = le.transform([subtype_map[p]])[0]
                    for img_path in selected_tiles[p]:
                        img = Image.open(img_path).convert("RGB")
                        tensor = transform(img).unsqueeze(0).to(device)
                        out = model(tensor)
                        features.append(out[0].cpu().numpy())
                        labels.append(label_encoded)
                        patient_ids.append(p)
            return np.array(features), np.array(labels), np.array(patient_ids)

        print("Extracting features for training set...")
        X_train, y_train, p_train = extract_features(train_patients)

        print("Extracting features for test set...")
        X_test, y_test, p_test = extract_features(test_patients)

        classifiers = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            "XGBoost": XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
            "MLP": MLPClassifier(hidden_layer_sizes=(512, 128), max_iter=500, random_state=42),
        }

        lb = LabelBinarizer()
        lb.fit(y_train)

        for clf_name, clf in classifiers.items():
            print(f"\n{'='*50}")
            print(f"Training {clf_name} on {encoder_name} features...")
            clf.fit(X_train, y_train)

            probs = clf.predict_proba(X_test)
            preds = np.argmax(probs, axis=1)

            accuracy = accuracy_score(y_test, preds)
            print(f"[{clf_name}] Accuracy: {accuracy * 100:.2f}%")

            try:
                y_true_bin = lb.transform(y_test)
                overall_auc = roc_auc_score(y_true_bin, probs, multi_class="ovr")
                print(f"[{clf_name}] Overall AUC (OvR): {overall_auc:.4f}")

                print(f"--- Per-Class AUC ({clf_name}) ---")
                for i, class_name in enumerate(classes):
                    if len(np.unique(y_true_bin[:, i])) > 1:
                        class_auc = roc_auc_score(y_true_bin[:, i], probs[:, i])
                        print(f"  {class_name}: {class_auc:.4f}")
                    else:
                        print(f"  {class_name}: N/A")
            except Exception as e:
                print(f"Could not compute AUC: {e}")

        del model
        torch.cuda.empty_cache()


def _load_uni2h(device):
    print("Loading UNI2-h model...")
    timm_kwargs = {
        'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
        'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667*2,
        'num_classes': 0, 'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU,
        'reg_tokens': 8, 'dynamic_img_size': True
    }
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    model.to(device)
    return model


def _load_resnet50(device):
    print("Loading ResNet50 (ImageNet)...")
    model = timm.create_model("resnet50", pretrained=True, num_classes=0)
    model.to(device)
    return model


if __name__ == "__main__":
    main()
