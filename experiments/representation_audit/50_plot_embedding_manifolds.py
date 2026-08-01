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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import umap

class ResNetSpatialPatchPredictor(nn.Module):
    def __init__(self, morph_dim=20, embed_dim=1536):
        super().__init__()
        self.encoder = timm.create_model('resnet18', pretrained=True, in_chans=4, features_only=True)
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

    def forward(self, spatial_4ch, morph):
        B = spatial_4ch.shape[0]
        features = self.encoder(spatial_4ch)
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

class UnifiedAblationDataset(Dataset):
    def __init__(self, meta_df, morph_df, spatial_dir, transform):
        self.meta_df = meta_df.reset_index(drop=True)
        self.morph_df = morph_df
        self.spatial_dir = spatial_dir
        self.transform = transform
        self.label_map = {label: i for i, label in enumerate(sorted(meta_df['label'].unique()))}
        
    def __len__(self):
        return len(self.meta_df)
    
    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        img_path = row['image_path']
        img_name = os.path.basename(img_path).replace('.png', '')
        
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        
        spatial_path = os.path.join(self.spatial_dir, f"{img_name}.npz")
        try:
            spatial_map = np.load(spatial_path)['map']
            spatial_map = spatial_map.astype(np.float32) / 255.0
            spatial_map = torch.tensor(spatial_map).permute(2, 0, 1)
        except:
            spatial_map = torch.zeros((5, 512, 512), dtype=torch.float32)
            
        if spatial_map.shape[0] == 5:
            spatial_4ch = spatial_map[[0, 1, 2, 4], :, :]
        else:
            spatial_4ch = torch.zeros((4, 512, 512), dtype=torch.float32)
            
        try:
            morph = self.morph_df.loc[img_name].values.astype(np.float32)
            morph = torch.tensor(morph)
        except:
            morph = torch.zeros(20, dtype=torch.float32)
            
        label = self.label_map[row['label']]
        return img_tensor, spatial_4ch, morph, label

def forward_up_to_block(model, x, block_idx):
    x = model.patch_embed(x)
    x = model._pos_embed(x)
    x = model.norm_pre(x)
    for i in range(block_idx + 1):
        x = model.blocks[i](x)
    return x

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
    MORPH_PATH = r"data/processed/classification/tcga_subtypes/features/morphology_with_counts.parquet"
    WEIGHTS_PATH = r"artifacts/models/adapters/sweep_weights/adapter_layer_12.pth"
    
    timm_kwargs = {
        'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
        'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667*2,
        'num_classes': 0, 'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 'dynamic_img_size': True
    }
    
    frozen_uni = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    frozen_uni.eval().to(device)
    transform = create_transform(**resolve_data_config(frozen_uni.pretrained_cfg, model=frozen_uni))
    
    meta_df = pd.read_csv(META_PATH)
    morph_df = pd.read_parquet(MORPH_PATH)
    
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(meta_df, test_size=0.2, stratify=meta_df['label'], random_state=42)
    
    test_dataset = UnifiedAblationDataset(test_df, morph_df, SPATIAL_DIR, transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    adapter = ResNetSpatialPatchPredictor(morph_dim=20).to(device)
    adapter.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    adapter.eval()
    
    layer_idx = 11 # Block 11 = Layer 12
    
    true_embeddings = []
    synth_embeddings = []
    labels_list = []
    
    print("Extracting Test Set Embeddings...")
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for img, spatial_4ch, morph, labels in tqdm(test_loader):
                img, spatial_4ch, morph = img.to(device), spatial_4ch.to(device), morph.to(device)
                
                # True Layer 12
                true_seq = forward_up_to_block(frozen_uni, img, layer_idx)
                true_pooled = true_seq.mean(dim=1).cpu().numpy()
                
                # Synthetic Layer 12
                synth_seq = adapter(spatial_4ch, morph)
                synth_pooled = synth_seq.mean(dim=1).cpu().numpy()
                
                true_embeddings.extend(true_pooled)
                synth_embeddings.extend(synth_pooled)
                labels_list.extend(labels.cpu().numpy())
                
    true_embeddings = np.array(true_embeddings)
    synth_embeddings = np.array(synth_embeddings)
    labels_list = np.array(labels_list)
    
    # Create combined dataset for PCA/UMAP
    X_combined = np.vstack((true_embeddings, synth_embeddings))
    y_combined = np.concatenate((labels_list, labels_list))
    source_labels = np.array(["True"] * len(true_embeddings) + ["Synthetic"] * len(synth_embeddings))
    
    print("Running PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_combined)
    
    print("Running UMAP...")
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = umap_reducer.fit_transform(X_combined)
    
    # Plotting
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    palette = {"True": "blue", "Synthetic": "red"}
    
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=source_labels, style=y_combined, 
                    palette=palette, alpha=0.6, ax=axes[0])
    axes[0].set_title("PCA: True vs Synthetic Manifold (Layer 12)")
    
    sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=source_labels, style=y_combined, 
                    palette=palette, alpha=0.6, ax=axes[1])
    axes[1].set_title("UMAP: True vs Synthetic Manifold (Layer 12)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "manifold_plot_layer12.png"), dpi=300)
    print(f"Saved manifold plot to {FIGURES_DIR}")

if __name__ == "__main__":
    main()
