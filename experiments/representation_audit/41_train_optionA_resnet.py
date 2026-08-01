import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import timm
import torch.nn.functional as F

class ResNetSpatialPatchPredictor(nn.Module):
    def __init__(self, morph_dim=16, embed_dim=1536):
        super().__init__()
        # Use first 3 channels (Tumor, Immune, Stroma)
        self.encoder = timm.create_model('resnet18', pretrained=True, features_only=True)
        # resnet18 features_only returns a list of feature maps. 
        # The final one (index 4) has shape [B, 512, H/32, W/32].
        # For 512x512 input, H/32 = 16, so output is [B, 512, 16, 16].
        
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
        # Get final feature map
        features = self.encoder(spatial_3ch)
        x_sp = features[-1] # [B, 512, 16, 16]
        
        # Patch Decoding
        morph_expanded = morph.view(B, -1, 1, 1).expand(-1, -1, 16, 16)
        x_patches = torch.cat([x_sp, morph_expanded], dim=1)
        patches_out = self.patch_decoder(x_patches) # [B, 1536, 16, 16]
        patches_seq = patches_out.flatten(2).transpose(1, 2) # [B, 256, 1536]
        
        # Prefix Decoding
        x_sp_global = x_sp.mean(dim=[2, 3]) # [B, 512]
        x_prefix = torch.cat([x_sp_global, morph], dim=1)
        prefix_out = self.prefix_decoder(x_prefix)
        prefix_seq = prefix_out.view(B, 9, 1536) # [B, 9, 1536]
        
        full_seq = torch.cat([prefix_seq, patches_seq], dim=1) # [B, 265, 1536]
        return full_seq

class IntermediateDataset(Dataset):
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
        
        # Load Spatial Map (5 channels)
        spatial_path = os.path.join(self.spatial_dir, f"{img_name}.npz")
        try:
            spatial_map = np.load(spatial_path)['map']
            spatial_map = spatial_map.astype(np.float32) / 255.0
            spatial_map = torch.tensor(spatial_map).permute(2, 0, 1)
        except:
            spatial_map = torch.zeros((5, 512, 512), dtype=torch.float32)
            
        # Slice to first 3 channels (Tumor, Immune, Stroma)
        spatial_3ch = spatial_map[:3, :, :]
        
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
            
        return spatial_3ch, morph, target

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
    
    meta_df = pd.read_csv(META_PATH)
    morph_df = pd.read_parquet(MORPH_PATH)
    
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(meta_df, test_size=0.2, stratify=meta_df['label'], random_state=42)
    
    batch_size = 64
    epochs = 20
    learning_rate = 1e-4
    layers = [1, 3, 6, 12, 23]
    
    for layer in layers:
        print(f"\n{'='*50}\nTraining ResNet Predictor for Layer {layer}\n{'='*50}")
        
        train_dataset = IntermediateDataset(train_df, morph_df, SPATIAL_DIR, DATA_DIR, layer)
        val_dataset = IntermediateDataset(val_df, morph_df, SPATIAL_DIR, DATA_DIR, layer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        model = ResNetSpatialPatchPredictor().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        save_path = os.path.join(MODEL_DIR, f"resnet_predictor_layer{layer}.pth")
        
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for spatial_3ch, morph, target in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
                spatial_3ch, morph, target = spatial_3ch.to(device), morph.to(device), target.to(device)
                
                optimizer.zero_grad()
                pred = model(spatial_3ch, morph)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * spatial_3ch.size(0)
                
            train_loss /= len(train_loader.dataset)
            
            model.eval()
            val_loss = 0.0
            cossim_sum = 0.0
            
            with torch.no_grad():
                for spatial_3ch, morph, target in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                    spatial_3ch, morph, target = spatial_3ch.to(device), morph.to(device), target.to(device)
                    pred = model(spatial_3ch, morph)
                    loss = criterion(pred, target)
                    val_loss += loss.item() * spatial_3ch.size(0)
                    
                    pred_flat = pred.flatten(1)
                    target_flat = target.flatten(1)
                    cossim = F.cosine_similarity(pred_flat, target_flat, dim=1).sum().item()
                    cossim_sum += cossim
                    
            val_loss /= len(val_loader.dataset)
            val_cossim = cossim_sum / len(val_loader.dataset)
            
            print(f"Epoch {epoch} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | Val CosSim: {val_cossim:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                print("  -> Saved best model!")

if __name__ == "__main__":
    main()
