import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch.nn.functional as F
import joblib

class ResNetSpatialPatchPredictor(nn.Module):
    def __init__(self, morph_dim=16, embed_dim=1536):
        super().__init__()
        self.encoder = timm.create_model('resnet18', pretrained=False, features_only=True)
        self.patch_decoder = nn.Sequential(
            nn.Conv2d(512 + morph_dim, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
            nn.Conv2d(1024, embed_dim, kernel_size=1)
        )
        self.prefix_decoder = nn.Sequential(
            nn.Linear(512 + morph_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 9 * embed_dim)
        )

    def forward(self, spatial_3ch, morph):
        B = spatial_3ch.shape[0]
        features = self.encoder(spatial_3ch)
        x_sp = features[-1]
        
        morph_expanded = morph.view(B, -1, 1, 1).expand(-1, -1, 16, 16)
        x_patches = torch.cat([x_sp, morph_expanded], dim=1)
        patches_out = self.patch_decoder(x_patches)
        patches_seq = patches_out.flatten(2).transpose(1, 2)
        
        x_sp_global = x_sp.mean(dim=[2, 3])
        x_prefix = torch.cat([x_sp_global, morph], dim=1)
        prefix_out = self.prefix_decoder(x_prefix)
        prefix_seq = prefix_out.view(B, 9, 1536)
        
        full_seq = torch.cat([prefix_seq, patches_seq], dim=1)
        return full_seq

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

    def predict_proba(self, X_t):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

class UnifiedAuditDataset(Dataset):
    def __init__(self, meta_df, morph_df, spatial_dir, features_dir, target_layer, transform):
        self.meta_df = meta_df.reset_index(drop=True)
        self.morph_df = morph_df
        self.spatial_dir = spatial_dir
        self.features_dir = features_dir
        self.target_layer = target_layer
        self.transform = transform
        self.label_map = {label: i for i, label in enumerate(sorted(meta_df['label'].unique()))}
        
    def __len__(self):
        return len(self.meta_df)
    
    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        img_path = row['image_path']
        img_name = os.path.basename(img_path).replace('.png', '')
        
        # Original Image
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        
        # Spatial Map
        spatial_path = os.path.join(self.spatial_dir, f"{img_name}.npz")
        try:
            spatial_map = np.load(spatial_path)['map']
            spatial_map = spatial_map.astype(np.float32) / 255.0
            spatial_map = torch.tensor(spatial_map).permute(2, 0, 1)
        except:
            spatial_map = torch.zeros((5, 512, 512), dtype=torch.float32)
            
        spatial_3ch = spatial_map[:3, :, :]
            
        # Morphology
        try:
            morph = self.morph_df.loc[img_name].values.astype(np.float32)
            morph = torch.tensor(morph)
        except:
            morph = torch.zeros(16, dtype=torch.float32)
            
        # True Target
        feat_path = os.path.join(self.features_dir, f"{img_name}.pt")
        try:
            feats = torch.load(feat_path, map_location='cpu', weights_only=True)
            target = feats[self.target_layer].float()
        except:
            target = torch.zeros((265, 1536), dtype=torch.float32)
            
        label = self.label_map[row['label']]
        return img_tensor, spatial_3ch, morph, target, label

def compute_spearman(pred, target):
    pred = pred.reshape(pred.size(0), -1)
    target = target.reshape(target.size(0), -1)
    rank_p = torch.argsort(torch.argsort(pred, dim=1), dim=1).float()
    rank_t = torch.argsort(torch.argsort(target, dim=1), dim=1).float()
    p_mean = rank_p.mean(dim=1, keepdim=True)
    t_mean = rank_t.mean(dim=1, keepdim=True)
    p_std = rank_p.std(dim=1, keepdim=True)
    t_std = rank_t.std(dim=1, keepdim=True)
    p_norm = (rank_p - p_mean) / p_std
    t_norm = (rank_t - t_mean) / t_std
    corr = (p_norm * t_norm).mean(dim=1)
    return corr.cpu().numpy()

def inject_forward(model, x, start_block_idx):
    for idx in range(start_block_idx, len(model.blocks)):
        x = model.blocks[idx](x)
    x = model.norm(x)
    return model.forward_head(x)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    DATA_DIR = r"data/misc/tcga_10k_cached_tensors"
    META_PATH = r"data/processed/classification/tcga_subtypes/manifests/legacy_10k_samples.csv"
    MODEL_DIR = r"artifacts/runs/legacy_representation_audit/models"
    METRICS_DIR = r"artifacts/runs/legacy_representation_audit/metrics"
    FIGURES_DIR = r"artifacts/runs/legacy_representation_audit/figures"
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    SPATIAL_DIR = r"data/processed/generator/spatial_maps"
    MORPH_PATH = r"data/processed/generator/morphology_features/morphology_standardized.parquet"
    CLASSIFIER_PATH = r"artifacts/models/downstream/mlp_uni2h.joblib"
    
    print("Loading Foundation Model...")
    timm_kwargs = {
        'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
        'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667*2,
        'num_classes': 0, 'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 'dynamic_img_size': True
    }
    uni = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    uni.eval().to(device)
    transform = create_transform(**resolve_data_config(uni.pretrained_cfg, model=uni))
    
    classifier = joblib.load(CLASSIFIER_PATH)
    classifier.device = device
    classifier.model.to(device)
    classifier.model.eval()
    
    meta_df = pd.read_csv(META_PATH)
    morph_df = pd.read_parquet(MORPH_PATH)
    
    from sklearn.model_selection import train_test_split
    _, test_df = train_test_split(meta_df, test_size=0.2, stratify=meta_df['label'], random_state=42)
    
    layers = [1, 3, 6, 12, 23]
    
    # Pre-compute real pathways once
    real_preds = []
    true_labels = []
    
    dummy_dataset = UnifiedAuditDataset(test_df, morph_df, SPATIAL_DIR, DATA_DIR, 1, transform)
    dummy_loader = DataLoader(dummy_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print("Computing True ViT Pathways on 20% Test Set...")
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for img, _, _, _, label in tqdm(dummy_loader, desc="Real Pathway"):
                img = img.to(device)
                true_emb = uni(img)
                probs = classifier.predict_proba(true_emb)
                preds = np.argmax(probs, axis=1)
                real_preds.extend(preds)
                true_labels.extend(label.numpy())
                
    real_preds = np.array(real_preds)
    true_labels = np.array(true_labels)
    
    results = []
    
    for layer in layers:
        print(f"\nEvaluating ResNet Predicted Layer {layer}...")
        
        predictor = ResNetSpatialPatchPredictor().to(device)
        pred_path = os.path.join(MODEL_DIR, f"resnet_predictor_layer{layer}.pth")
        if not os.path.exists(pred_path):
            print(f"Skipping layer {layer}, predictor not found.")
            continue
            
        predictor.load_state_dict(torch.load(pred_path, map_location=device))
        predictor.eval()
        
        dataset = UnifiedAuditDataset(test_df, morph_df, SPATIAL_DIR, DATA_DIR, layer, transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        
        layer_preds = []
        spearmans = []
        cossims = []
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                for _, spatial_3ch, morph, target, _ in tqdm(loader, desc=f"Layer {layer} Audit"):
                    spatial_3ch, morph, target = spatial_3ch.to(device), morph.to(device), target.to(device)
                    
                    pred_seq = predictor(spatial_3ch, morph)
                    
                    # Compute Spearman
                    batch_sp = compute_spearman(pred_seq, target)
                    spearmans.extend(batch_sp)
                    
                    # Compute Cosine
                    pred_flat = pred_seq.flatten(1)
                    target_flat = target.flatten(1)
                    batch_cos = F.cosine_similarity(pred_flat, target_flat, dim=1).cpu().numpy()
                    cossims.extend(batch_cos)
                    
                    # Forward Prop
                    injected_emb = inject_forward(uni, pred_seq, layer + 1)
                    probs = classifier.predict_proba(injected_emb)
                    preds = np.argmax(probs, axis=1)
                    layer_preds.extend(preds)
                    
        layer_preds = np.array(layer_preds)
        agreement = np.mean(layer_preds == real_preds)
        accuracy = np.mean(layer_preds == true_labels)
        avg_spearman = np.mean(spearmans)
        avg_cossim = np.mean(cossims)
        
        print(f"Layer {layer} -> CosSim: {avg_cossim:.4f} | Spearman: {avg_spearman:.4f} | Agreement: {agreement*100:.2f}% | Downstream Accuracy: {accuracy*100:.2f}%")
        results.append({'Layer': layer, 'CosSim': avg_cossim, 'Spearman': avg_spearman, 'Agreement': agreement, 'Accuracy': accuracy})
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(METRICS_DIR, "resnet_forward_prop_audit_results.csv"), index=False)
    print("\nFinal Results:")
    print(res_df)

if __name__ == "__main__":
    main()
