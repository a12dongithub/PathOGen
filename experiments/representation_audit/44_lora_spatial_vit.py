import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from peft import LoraConfig, get_peft_model
import joblib

class ResNetSpatialPatchPredictor(nn.Module):
    def __init__(self, morph_dim=16, embed_dim=1536):
        super().__init__()
        self.encoder = timm.create_model('resnet18', pretrained=True, features_only=True)
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

class DownstreamClassificationDataset(Dataset):
    def __init__(self, meta_df, morph_df, spatial_dir):
        self.meta_df = meta_df.reset_index(drop=True)
        self.morph_df = morph_df
        self.spatial_dir = spatial_dir
        self.label_map = {label: i for i, label in enumerate(sorted(meta_df['label'].unique()))}
        
    def __len__(self):
        return len(self.meta_df)
    
    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        img_path = row['image_path']
        img_name = os.path.basename(img_path).replace('.png', '')
        
        spatial_path = os.path.join(self.spatial_dir, f"{img_name}.npz")
        try:
            spatial_map = np.load(spatial_path)['map']
            spatial_map = spatial_map.astype(np.float32) / 255.0
            spatial_map = torch.tensor(spatial_map).permute(2, 0, 1)
        except:
            spatial_map = torch.zeros((5, 512, 512), dtype=torch.float32)
            
        spatial_3ch = spatial_map[:3, :, :]
            
        try:
            morph = self.morph_df.loc[img_name].values.astype(np.float32)
            morph = torch.tensor(morph)
        except:
            morph = torch.zeros(16, dtype=torch.float32)
            
        label = self.label_map[row['label']]
        return spatial_3ch, morph, label

def inject_forward(model, x, start_block_idx):
    # Depending on PEFT wrapping, model.blocks might be inside model.base_model.model
    # Let's find the blocks module
    if hasattr(model, 'base_model'):
        blocks = model.base_model.model.blocks
        norm = model.base_model.model.norm
        forward_head = model.base_model.model.forward_head
    else:
        blocks = model.blocks
        norm = model.norm
        forward_head = model.forward_head
        
    for idx in range(start_block_idx, len(blocks)):
        x = blocks[idx](x)
    x = norm(x)
    return forward_head(x)

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
    
    # Load MLP
    classifier = joblib.load(CLASSIFIER_PATH)
    classifier.device = device
    classifier.model.to(device)
    classifier.model.eval()
    
    meta_df = pd.read_csv(META_PATH)
    morph_df = pd.read_parquet(MORPH_PATH)
    
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(meta_df, test_size=0.2, stratify=meta_df['label'], random_state=42)
    
    train_dataset = DownstreamClassificationDataset(train_df, morph_df, SPATIAL_DIR)
    test_dataset = DownstreamClassificationDataset(test_df, morph_df, SPATIAL_DIR)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    layers = [1, 3, 6, 12, 23]
    
    timm_kwargs = {
        'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
        'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667*2,
        'num_classes': 0, 'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 'dynamic_img_size': True
    }
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["qkv", "proj", "fc1", "fc2"],
        lora_dropout=0.1,
        bias="none",
    )
    
    results = []
    
    for layer in layers:
        print(f"\n{'='*50}\nLoRA Alignment for Layer {layer}\n{'='*50}")
        
        # 1. Load fresh UNI-2h & wrap with LoRA
        uni = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        uni = get_peft_model(uni, lora_config)
        uni.to(device)
        uni.train()
        
        # 2. Load the pre-trained Adapter
        adapter = ResNetSpatialPatchPredictor().to(device)
        pred_path = os.path.join(MODEL_DIR, f"resnet_predictor_layer{layer}.pth")
        if os.path.exists(pred_path):
            adapter.load_state_dict(torch.load(pred_path, map_location=device))
            print(f"Loaded pretrained adapter for Layer {layer}.")
        else:
            print(f"WARNING: Pretrained adapter for Layer {layer} not found! Initializing randomly.")
            
        adapter.train()
        
        # Optimizer includes both Adapter and LoRA parameters
        params = list(adapter.parameters()) + list(uni.parameters())
        optimizer = optim.AdamW(params, lr=5e-5, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        epochs = 5 # Small number of epochs for alignment
        
        for epoch in range(1, epochs + 1):
            adapter.train()
            uni.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for spatial_3ch, morph, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
                spatial_3ch, morph, labels = spatial_3ch.to(device), morph.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.amp.autocast('cuda'):
                    pred_seq = adapter(spatial_3ch, morph)
                    injected_emb = inject_forward(uni, pred_seq, layer + 1)
                    logits = classifier.model(injected_emb)
                    loss = criterion(logits, labels)
                    
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * spatial_3ch.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
            train_acc = correct / total
            
            # Eval
            adapter.eval()
            uni.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    for spatial_3ch, morph, labels in test_loader:
                        spatial_3ch, morph, labels = spatial_3ch.to(device), morph.to(device), labels.to(device)
                        pred_seq = adapter(spatial_3ch, morph)
                        injected_emb = inject_forward(uni, pred_seq, layer + 1)
                        logits = classifier.model(injected_emb)
                        preds = torch.argmax(logits, dim=1)
                        val_correct += (preds == labels).sum().item()
                        val_total += labels.size(0)
            
            val_acc = val_correct / val_total
            print(f"Epoch {epoch} | Train Loss: {train_loss/total:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
            
        print(f"Final Val Acc for Layer {layer}: {val_acc*100:.2f}%")
        results.append({'Layer': layer, 'Aligned_Accuracy': val_acc})
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(METRICS_DIR, "lora_aligned_audit_results.csv"), index=False)
    print("\nFinal Alignment Results:")
    print(res_df)

if __name__ == "__main__":
    main()
