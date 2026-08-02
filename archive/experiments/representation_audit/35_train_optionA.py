import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import math

class OptionADataset(Dataset):
    def __init__(self, meta_df, morph_df, spatial_dir, features_dir, target_layer):
        self.meta_df = meta_df.reset_index(drop=True)
        self.morph_df = morph_df
        self.spatial_dir = spatial_dir
        self.features_dir = features_dir
        self.target_layer = target_layer
        
        # We need a label map to do stratified splitting, though we train on everything in the train_idx
        self.label_map = {label: i for i, label in enumerate(sorted(meta_df['label'].unique()))}
        
    def __len__(self):
        return len(self.meta_df)
    
    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        img_path = row['image_path']
        img_name = os.path.basename(img_path).replace('.png', '')
        
        # Load Spatial Map
        spatial_path = os.path.join(self.spatial_dir, f"{img_name}.npz")
        try:
            spatial_map = np.load(spatial_path)['map'] # [512, 512, 5]
            spatial_map = spatial_map.astype(np.float32) / 255.0
            spatial_map = torch.tensor(spatial_map).permute(2, 0, 1) # [5, 512, 512]
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
            target = feats[self.target_layer].float() # [265, 1536]
        except:
            target = torch.zeros((265, 1536), dtype=torch.float32)
            
        label = self.label_map[row['label']]
        return spatial_map, morph, target, label

class SpatialPatchPredictor(nn.Module):
    def __init__(self, in_channels=5, morph_dim=16, embed_dim=1536):
        super().__init__()
        
        # Spatial Encoder: 512x512 -> 256x256 -> 128x128 -> 64x64 -> 32x32 -> 16x16
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
        ) # Output: [B, 1024, 16, 16]
        
        # Patch Decoder (Combines Spatial + Morph -> 16x16 patches)
        self.patch_decoder = nn.Sequential(
            nn.Conv2d(1024 + morph_dim, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
            nn.Conv2d(1024, embed_dim, kernel_size=1)
        ) # Output: [B, 1536, 16, 16]
        
        # Prefix Decoder (Combines Global pooled spatial + Morph -> 9 prefix tokens)
        self.prefix_decoder = nn.Sequential(
            nn.Linear(1024 + morph_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 9 * embed_dim) # 9 prefix tokens
        )

    def forward(self, spatial, morph):
        # spatial: [B, 5, 512, 512]
        # morph: [B, 16]
        B = spatial.shape[0]
        
        # Encode Spatial
        x_sp = self.encoder(spatial) # [B, 1024, 16, 16]
        
        # Expand morphology to spatial grid
        morph_expanded = morph.view(B, -1, 1, 1).expand(-1, -1, 16, 16)
        
        # Combine for patches
        x_patches = torch.cat([x_sp, morph_expanded], dim=1)
        patches_out = self.patch_decoder(x_patches) # [B, 1536, 16, 16]
        
        # Reshape patches to sequence [B, 256, 1536]
        patches_seq = patches_out.flatten(2).transpose(1, 2)
        
        # Combine for prefix tokens
        x_sp_global = x_sp.mean(dim=[2, 3]) # [B, 1024]
        x_prefix = torch.cat([x_sp_global, morph], dim=1) # [B, 1040]
        prefix_out = self.prefix_decoder(x_prefix) # [B, 9*1536]
        prefix_seq = prefix_out.view(B, 9, 1536) # [B, 9, 1536]
        
        # Final sequence
        full_seq = torch.cat([prefix_seq, patches_seq], dim=1) # [B, 265, 1536]
        return full_seq

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
    
    # Stratified Split (80/20) based on labels
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(meta_df, test_size=0.2, stratify=meta_df['label'], random_state=42)
    
    print(f"Training Option A Predictors on {len(train_df)} samples, Testing on {len(test_df)}")
    
    layers = [1, 3, 6, 12, 23]
    
    for target_layer in layers:
        print(f"\n=============================================")
        print(f" Training Predictor for Layer {target_layer}")
        print(f"=============================================")
        
        train_dataset = OptionADataset(train_df, morph_df, SPATIAL_DIR, DATA_DIR, target_layer)
        test_dataset = OptionADataset(test_df, morph_df, SPATIAL_DIR, DATA_DIR, target_layer)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        model = SpatialPatchPredictor().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        
        epochs = 20
        best_val_loss = float('inf')
        save_path = os.path.join(MODEL_DIR, f"optionA_predictor_layer{target_layer}.pth")
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for spatial, morph, target, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
                spatial, morph, target = spatial.to(device), morph.to(device), target.to(device)
                
                optimizer.zero_grad()
                pred = model(spatial, morph)
                
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            cos_sims = []
            with torch.no_grad():
                for spatial, morph, target, _ in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    spatial, morph, target = spatial.to(device), morph.to(device), target.to(device)
                    pred = model(spatial, morph)
                    loss = criterion(pred, target)
                    val_loss += loss.item()
                    
                    # Compute Cosine Similarity (flatten sequence)
                    t_flat = target.reshape(target.shape[0], -1)
                    p_flat = pred.reshape(pred.shape[0], -1)
                    
                    t_norm = t_flat / torch.norm(t_flat, dim=1, keepdim=True).clamp(min=1e-8)
                    p_norm = p_flat / torch.norm(p_flat, dim=1, keepdim=True).clamp(min=1e-8)
                    sim = (t_norm * p_norm).sum(dim=1)
                    cos_sims.extend(sim.cpu().numpy())
                    
            val_loss /= len(test_loader)
            mean_cos = np.mean(cos_sims)
            
            print(f"Epoch {epoch+1:02d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | Val CosSim: {mean_cos:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                print("  -> Saved best model!")
                
if __name__ == "__main__":
    main()
