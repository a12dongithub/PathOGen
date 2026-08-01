import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

class OptionBDataset(Dataset):
    def __init__(self, meta_df, morph_df, spatial_dir, features_dir, target_layer):
        self.meta_df = meta_df.reset_index(drop=True)
        self.morph_df = morph_df
        self.spatial_dir = spatial_dir
        self.features_dir = features_dir
        self.target_layer = target_layer
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
            full_seq = feats[self.target_layer].float() # [265, 1536]
        except:
            full_seq = torch.zeros((265, 1536), dtype=torch.float32)
            
        # COMPUTE OPTION B TARGETS (CLS + 3 Regional Tokens)
        cls_token = full_seq[0] # [1536] (prefix[0] is CLS in timm)
        patches = full_seq[9:] # [256, 1536]
        patches_spatial = patches.view(16, 16, 1536).permute(2, 0, 1).unsqueeze(0) # [1, 1536, 16, 16]
        
        # Downsample spatial mask to 16x16
        mask_16 = F.adaptive_avg_pool2d(spatial_map.unsqueeze(0), (16, 16)) # [1, 5, 16, 16]
        
        regional_tokens = []
        for c in range(3): # Ch0=Tumor, Ch1=Immune, Ch2=Stroma
            weight = mask_16[0, c, :, :].flatten() # [256]
            weight_sum = weight.sum()
            if weight_sum > 1e-6:
                w_norm = weight / weight_sum
                # Weighted sum of patches
                reg_tok = (patches * w_norm.unsqueeze(1)).sum(dim=0)
            else:
                # If region doesn't exist, just use global mean of patches to avoid zeros
                reg_tok = patches.mean(dim=0)
            regional_tokens.append(reg_tok)
            
        # Target is [CLS, Tumor, Immune, Stroma]
        target = torch.cat([cls_token] + regional_tokens, dim=0) # [6144]
        
        label = self.label_map[row['label']]
        return spatial_map, morph, target, label

class OptionBPredictor(nn.Module):
    def __init__(self, in_channels=5, morph_dim=16, embed_dim=1536):
        super().__init__()
        
        # Simple spatial CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Target size is 4 tokens (CLS + 3 regions) -> 6144D
        self.mlp = nn.Sequential(
            nn.Linear(256 + morph_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4 * embed_dim)
        )

    def forward(self, spatial, morph):
        x_sp = self.cnn(spatial).flatten(1)
        x_in = torch.cat([x_sp, morph], dim=1)
        return self.mlp(x_in)

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
    train_df, test_df = train_test_split(meta_df, test_size=0.2, stratify=meta_df['label'], random_state=42)
    
    print(f"Training Option B Predictors on {len(train_df)} samples, Testing on {len(test_df)}")
    
    layers = [1, 3, 6, 12, 23]
    
    for target_layer in layers:
        print(f"\n=============================================")
        print(f" Training Predictor B for Layer {target_layer}")
        print(f"=============================================")
        
        train_dataset = OptionBDataset(train_df, morph_df, SPATIAL_DIR, DATA_DIR, target_layer)
        test_dataset = OptionBDataset(test_df, morph_df, SPATIAL_DIR, DATA_DIR, target_layer)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
        
        model = OptionBPredictor().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        
        epochs = 20
        best_val_loss = float('inf')
        save_path = os.path.join(MODEL_DIR, f"optionB_predictor_layer{target_layer}.pth")
        
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
                    
                    t_norm = target / torch.norm(target, dim=1, keepdim=True).clamp(min=1e-8)
                    p_norm = pred / torch.norm(pred, dim=1, keepdim=True).clamp(min=1e-8)
                    sim = (t_norm * p_norm).sum(dim=1) / 4.0 # Average sim across the 4 tokens
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
