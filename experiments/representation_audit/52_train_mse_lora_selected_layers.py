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
import torch.nn.functional as F
import joblib

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

class PyTorchMLPClassifier:
    def __init__(self, input_dim, hidden_layer_sizes=(512, 128), max_iter=100, lr=1e-3, device="cuda"):
        self.input_dim = input_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.lr = lr
        self.device = device
        self.model = None

    def predict_proba(self, X_t):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

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

def extract_target_features(model, x, block_idx):
    if hasattr(model, 'base_model'):
        blocks = model.base_model.model.blocks
        patch_embed = model.base_model.model.patch_embed
        pos_embed = getattr(model.base_model.model, 'pos_embed', None)
        cls_token = getattr(model.base_model.model, 'cls_token', None)
        reg_token = getattr(model.base_model.model, 'reg_token', None)
    else:
        blocks = model.blocks
        patch_embed = model.patch_embed
        pos_embed = getattr(model, 'pos_embed', None)
        cls_token = getattr(model, 'cls_token', None)
        reg_token = getattr(model, 'reg_token', None)

    x = patch_embed(x)
    # Flatten spatial dimensions [B, H, W, C] -> [B, H*W, C]
    if x.ndim == 4:
        x = x.flatten(1, 2)
        
    if pos_embed is not None:
        x = x + pos_embed
        
    tokens = []
    if cls_token is not None:
        tokens.append(cls_token.expand(x.shape[0], -1, -1))
    if reg_token is not None:
        tokens.append(reg_token.expand(x.shape[0], -1, -1))
        
    if len(tokens) > 0:
        x = torch.cat(tokens + [x], dim=1)
        
    for i in range(block_idx):
        x = blocks[i](x)
    return x

def inject_forward(model, x, start_block_idx):
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

def collate_fn(batch):
    img_tensors = torch.stack([b[0] for b in batch])
    spatial_4chs = torch.stack([b[1] for b in batch])
    morphs = torch.stack([b[2] for b in batch])
    labels = torch.tensor([b[3] for b in batch])
    return img_tensors, spatial_4chs, morphs, labels

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
    WEIGHTS_DIR = r"artifacts/models/adapters/high_acc_weights"
    CLASSIFIER_PATH = r"artifacts/models/downstream/mlp_uni2h.joblib"
    
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
    
    classifier = joblib.load(CLASSIFIER_PATH)
    classifier.device = device
    classifier.model.to(device)
    classifier.model.eval()
    
    meta_df = pd.read_csv(META_PATH)
    morph_df = pd.read_parquet(MORPH_PATH)
    
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(meta_df, test_size=0.2, stratify=meta_df['label'], random_state=42)
    
    train_dataset = UnifiedAblationDataset(train_df, morph_df, SPATIAL_DIR, transform)
    test_dataset = UnifiedAblationDataset(test_df, morph_df, SPATIAL_DIR, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)

    print("Pre-computing True ViT Pathways on Test Set...")
    real_preds = []
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for img, _, _, _ in tqdm(test_loader, desc="Real Pathway"):
                img = img.to(device)
                true_emb = frozen_uni(img)
                probs = classifier.predict_proba(true_emb)
                preds = np.argmax(probs, axis=1)
                real_preds.extend(preds)
    real_preds = np.array(real_preds)
    
    # 5 is layer 6, 11 is layer 12, 17 is layer 18, 23 is layer 24
    layers_to_sweep = [5, 11, 17, 23]
    results = {}
    
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    for layer_idx in layers_to_sweep:
        print(f"\n{'='*50}\nTraining Pipeline for Block Index {layer_idx} (Layer {layer_idx+1})\n{'='*50}")
        
        adapter = ResNetSpatialPatchPredictor(morph_dim=20).to(device)
        
        # --- PHASE 1: MSE Pre-training ---
        print("\n--- PHASE 1: MSE Pre-training (10 Epochs) ---")
        mse_optimizer = optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=1e-4)
        mse_criterion = nn.MSELoss()
        
        adapter.train()
        for epoch in range(1, 11):
            train_loss = 0.0
            for img, spatial_4ch, morph, _ in tqdm(train_loader, desc=f"MSE Epoch {epoch}/10"):
                img = img.to(device)
                spatial_4ch = spatial_4ch.to(device)
                morph = morph.to(device)
                
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        target_seq = extract_target_features(frozen_uni, img, layer_idx)
                        
                mse_optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    pred_seq = adapter(spatial_4ch, morph)
                    loss = mse_criterion(pred_seq, target_seq.detach())
                loss.backward()
                mse_optimizer.step()
                train_loss += loss.item()
            print(f"MSE Epoch {epoch} Loss: {train_loss / len(train_loader):.4f}")
            
        torch.save(adapter.state_dict(), os.path.join(WEIGHTS_DIR, f"adapter_mse_layer_{layer_idx+1}.pth"))
        
        # --- PHASE 2: End-to-End LoRA Fine-Tuning ---
        print("\n--- PHASE 2: LoRA Fine-tuning (5 Epochs) ---")
        lora_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["qkv", "proj", "fc1", "fc2"],
            lora_dropout=0.1, bias="none"
        )
        lora_uni = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        lora_uni = get_peft_model(lora_uni, lora_config)
        lora_uni.to(device)
        
        # Freeze ResNet backbone, unfreeze projection heads for fine-tuning flexibility
        for param in adapter.encoder.parameters():
            param.requires_grad = False
            
        lora_uni.train()
        adapter.train()
        
        params = list(filter(lambda p: p.requires_grad, adapter.parameters())) + list(lora_uni.parameters())
        ce_optimizer = optim.AdamW(params, lr=5e-5, weight_decay=1e-4)
        ce_criterion = nn.CrossEntropyLoss()
        
        for epoch in range(1, 6):
            for img, spatial_4ch, morph, labels in tqdm(train_loader, desc=f"LoRA Epoch {epoch}/5"):
                spatial_4ch = spatial_4ch.to(device)
                morph = morph.to(device)
                labels = labels.to(device)
                
                ce_optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    pred_seq = adapter(spatial_4ch, morph)
                    injected_emb = inject_forward(lora_uni, pred_seq, layer_idx)
                    logits = classifier.model(injected_emb)
                    loss = ce_criterion(logits, labels)
                loss.backward()
                ce_optimizer.step()
                
        torch.save(adapter.state_dict(), os.path.join(WEIGHTS_DIR, f"adapter_lora_layer_{layer_idx+1}.pth"))
        
        # --- PHASE 3: Evaluation ---
        print("\n--- PHASE 3: Evaluation ---")
        adapter.eval()
        lora_uni.eval()
        frozen_injected_preds = []
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                for _, spatial_4ch, morph, _ in test_loader:
                    spatial_4ch = spatial_4ch.to(device)
                    morph = morph.to(device)
                    pred_seq = adapter(spatial_4ch, morph)
                    injected_emb = inject_forward(frozen_uni, pred_seq, layer_idx)
                    preds = np.argmax(classifier.predict_proba(injected_emb), axis=1)
                    frozen_injected_preds.extend(preds)
                    
        acc = np.mean(np.array(frozen_injected_preds) == real_preds) * 100
        print(f"Layer {layer_idx+1} Top-1 Agreement: {acc:.2f}%")
        results[f"Layer {layer_idx+1}"] = acc
        
        pd.DataFrame(list(results.items()), columns=['Layer', 'Agreement']).to_csv(
            os.path.join(METRICS_DIR, "high_acc_selected_layers_results.csv"), index=False
        )

    print("\nFINAL RESULTS:")
    for k, v in results.items():
        print(f"{k}: {v:.2f}%")

if __name__ == "__main__":
    main()
