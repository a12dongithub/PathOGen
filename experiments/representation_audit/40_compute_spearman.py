import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm



class SpearmanDataset(Dataset):
    def __init__(self, meta_df, morph_df, spatial_dir, features_dir, target_layer):
        self.meta_df = meta_df.reset_index(drop=True)
        self.morph_df = morph_df
        self.spatial_dir = spatial_dir
        self.features_dir = features_dir
        self.target_layer = target_layer
        
    def __len__(self):
        return len(self.meta_df)
    
    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        img_path = row['image_path']
        img_name = os.path.basename(img_path).replace('.png', '')
        
        # Load Spatial Map
        spatial_path = os.path.join(self.spatial_dir, f"{img_name}.npz")
        try:
            spatial_map = np.load(spatial_path)['map']
            spatial_map = spatial_map.astype(np.float32) / 255.0
            spatial_map = torch.tensor(spatial_map).permute(2, 0, 1)
        except:
            spatial_map = torch.zeros((5, 512, 512), dtype=torch.float32)
            
        # Load Morphology
        try:
            morph = self.morph_df.loc[img_name].values.astype(np.float32)
            morph = torch.tensor(morph)
        except:
            morph = torch.zeros(16, dtype=torch.float32)
            
        # Load Target Feature (Sequence [265, 1536])
        feat_path = os.path.join(self.features_dir, f"{img_name}.pt")
        try:
            feats = torch.load(feat_path, map_location='cpu', weights_only=True)
            target = feats[self.target_layer].float()
        except:
            target = torch.zeros((265, 1536), dtype=torch.float32)
            
        return spatial_map, morph, target

def compute_spearman(pred, target):
    # Flatten sequences: [B, 265*1536]
    pred = pred.reshape(pred.size(0), -1)
    target = target.reshape(target.size(0), -1)
    
    # Ranks
    rank_p = torch.argsort(torch.argsort(pred, dim=1), dim=1).float()
    rank_t = torch.argsort(torch.argsort(target, dim=1), dim=1).float()
    
    # Standardize
    p_mean = rank_p.mean(dim=1, keepdim=True)
    t_mean = rank_t.mean(dim=1, keepdim=True)
    
    p_std = rank_p.std(dim=1, keepdim=True)
    t_std = rank_t.std(dim=1, keepdim=True)
    
    p_norm = (rank_p - p_mean) / p_std
    t_norm = (rank_t - t_mean) / t_std
    
    corr = (p_norm * t_norm).mean(dim=1)
    return corr.cpu().numpy()

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
    SPATIAL_DIR = r"data/processed/conditions/spatial_maps"
    MORPH_PATH = r"data/processed/conditions/morphology/standardized.parquet"
    
    meta_df = pd.read_csv(META_PATH)
    morph_df = pd.read_parquet(MORPH_PATH)
    
    from sklearn.model_selection import train_test_split
    _, test_df = train_test_split(meta_df, test_size=0.2, stratify=meta_df['label'], random_state=42)
    
    layers = [1, 3, 6, 12, 23]
    
    # We redefine SpatialPatchPredictor exactly to avoid import syntax error on '35_train...'
    class SpatialPatchPredictorInline(nn.Module):
        def __init__(self, in_channels=5, morph_dim=16, embed_dim=1536):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(1024), nn.ReLU(inplace=True)
            )
            self.patch_decoder = nn.Sequential(
                nn.Conv2d(1024 + morph_dim, 1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
                nn.Conv2d(1024, embed_dim, kernel_size=1)
            )
            self.prefix_decoder = nn.Sequential(
                nn.Linear(1024 + morph_dim, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 9 * embed_dim)
            )

        def forward(self, spatial, morph):
            B = spatial.shape[0]
            x_sp = self.encoder(spatial)
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
            
    print("Computing Spearman Correlations...")
    for layer in layers:
        dataset = SpearmanDataset(test_df, morph_df, SPATIAL_DIR, DATA_DIR, layer)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        
        predictor = SpatialPatchPredictorInline().to(device)
        pred_path = os.path.join(MODEL_DIR, f"optionA_predictor_layer{layer}.pth")
        if not os.path.exists(pred_path):
            continue
            
        predictor.load_state_dict(torch.load(pred_path, map_location=device))
        predictor.eval()
        
        spearmans = []
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                for spatial, morph, target in tqdm(loader, desc=f"Layer {layer}"):
                    spatial, morph, target = spatial.to(device), morph.to(device), target.to(device)
                    pred = predictor(spatial, morph)
                    batch_sp = compute_spearman(pred, target)
                    spearmans.extend(batch_sp)
                    
        print(f"Layer {layer} -> Average Spearman Correlation: {np.mean(spearmans):.4f}")

if __name__ == "__main__":
    main()
