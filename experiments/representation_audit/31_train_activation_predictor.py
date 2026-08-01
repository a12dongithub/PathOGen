import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
import joblib

# Paths
BENCHMARK_FEATURES_PATH = r"data/processed/classification/tcga_subtypes/embeddings/combined/benchmark_features.parquet"
MORPH_STATS_PATH = r"data/processed/generator/morphology_features/morphology_standardized.parquet"
SPATIAL_DIR = r"data/processed/generator/spatial_maps"
OUTPUT_DIR = r"artifacts/runs/activation_poc"

os.makedirs(OUTPUT_DIR, exist_ok=True)

class ActivationDataset(Dataset):
    def __init__(self, df, morph_df, spatial_dir):
        self.df = df.reset_index(drop=True)
        self.morph_df = morph_df
        self.spatial_dir = spatial_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        stem = Path(img_path).stem
        
        # Target: UNI-2h embedding (1536D)
        uni_emb = torch.tensor(row['uni2h'], dtype=torch.float32)
        
        # Input 1: Morphology (16D)
        morph = torch.tensor(self.morph_df.loc[stem].values, dtype=torch.float32)
        
        # Input 2: Spatial Map (5x512x512)
        spatial_path = os.path.join(self.spatial_dir, f"{stem}.npz")
        spatial_data = np.load(spatial_path)['map'] # (512, 512, 5) uint8
        spatial_data = spatial_data.astype(np.float32) / 255.0
        spatial_tensor = torch.from_numpy(spatial_data).permute(2, 0, 1) # (5, 512, 512)
        
        return spatial_tensor, morph, uni_emb, stem

class ActivationPredictor(nn.Module):
    def __init__(self, spatial_in_channels=5, morph_dim=16, target_dim=1536):
        super().__init__()
        
        # Spatial Branch: CNN [B, 5, 512, 512] -> [B, 512]
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(spatial_in_channels, 32, kernel_size=3, stride=2, padding=1), # 256
            nn.ReLU(),
            nn.MaxPool2d(2), # 128
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 64
            nn.ReLU(),
            nn.MaxPool2d(2), # 32
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 16
            nn.ReLU(),
            nn.MaxPool2d(2), # 8
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 4
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), # [B, 256, 1, 1]
            nn.Flatten() # [B, 256]
        )
        
        # Morphology Branch: [B, 16] -> [B, 64]
        self.morph_mlp = nn.Sequential(
            nn.Linear(morph_dim, 64),
            nn.ReLU()
        )
        
        # Fusion Head: [B, 256 + 64] -> [B, 1536]
        self.fusion_mlp = nn.Sequential(
            nn.Linear(256 + 64, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, target_dim)
        )
        
    def forward(self, spatial, morph):
        s_feat = self.spatial_cnn(spatial)
        m_feat = self.morph_mlp(morph)
        fused = torch.cat([s_feat, m_feat], dim=1)
        out = self.fusion_mlp(fused)
        return out

def main():
    print("Loading data...")
    df = pd.read_parquet(BENCHMARK_FEATURES_PATH)
    # Remove Normal-like if it's there
    df = df[df['label'] != 'BRCA.Normal-like'].copy()
    
    morph_df = pd.read_parquet(MORPH_STATS_PATH)
    
    # Filter df to only keep tiles that have a spatial map
    valid_stems = set([Path(p).stem for p in os.listdir(SPATIAL_DIR) if p.endswith('.npz')])
    df['stem'] = df['image_path'].apply(lambda x: Path(x).stem)
    df = df[df['stem'].isin(valid_stems)].copy()
    
    print(f"Total valid tiles: {len(df)}")
    
    # Group by patient
    df['patient_id'] = df['stem'].apply(lambda x: x.split('_')[0][:12])
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df['patient_id']))
    
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    
    print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")
    
    # Save test set details for later evaluation
    test_df[['image_path', 'stem', 'label', 'patient_id']].to_csv(os.path.join(OUTPUT_DIR, "test_set_metadata.csv"), index=False)
    
    train_dataset = ActivationDataset(train_df, morph_df, SPATIAL_DIR)
    test_dataset = ActivationDataset(test_df, morph_df, SPATIAL_DIR)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActivationPredictor().to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    mse_criterion = nn.MSELoss()
    cos_criterion = nn.CosineEmbeddingLoss()
    
    EPOCHS = 30
    best_val_loss = float('inf')
    
    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        # Use target=1 for CosineEmbeddingLoss to encourage alignment
        target_ones = torch.ones(32).to(device)
        
        for spatial, morph, target, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            spatial, morph, target = spatial.to(device), morph.to(device), target.to(device)
            
            optimizer.zero_grad()
            pred = model(spatial, morph)
            
            # Dynamic batch size for cosine loss targets
            curr_bs = pred.size(0)
            
            loss_mse = mse_criterion(pred, target)
            loss_cos = cos_criterion(pred, target, torch.ones(curr_bs).to(device))
            
            # Combine losses
            loss = loss_mse + 0.5 * loss_cos
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        val_cos = 0.0
        with torch.no_grad():
            for spatial, morph, target, _ in test_loader:
                spatial, morph, target = spatial.to(device), morph.to(device), target.to(device)
                pred = model(spatial, morph)
                
                curr_bs = pred.size(0)
                l_mse = mse_criterion(pred, target)
                l_cos = cos_criterion(pred, target, torch.ones(curr_bs).to(device))
                loss = l_mse + 0.5 * l_cos
                
                val_loss += loss.item()
                val_mse += l_mse.item()
                
                # compute raw cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(pred, target, dim=1).mean()
                val_cos += cos_sim.item()
                
        val_loss /= len(test_loader)
        val_mse /= len(test_loader)
        val_cos /= len(test_loader)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MSE: {val_mse:.4f} | Val CosSim: {val_cos:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_activation_predictor.pth"))
            print("  -> Saved best model!")

if __name__ == "__main__":
    main()
