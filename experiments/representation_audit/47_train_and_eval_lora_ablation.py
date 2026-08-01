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

class UnifiedAblationDataset(Dataset):
    def __init__(self, meta_df, morph_df, spatial_dir, features_dir, transform):
        self.meta_df = meta_df.reset_index(drop=True)
        self.morph_df = morph_df
        self.spatial_dir = spatial_dir
        self.features_dir = features_dir
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
            
        spatial_3ch = spatial_map[:3, :, :]
            
        try:
            morph = self.morph_df.loc[img_name].values.astype(np.float32)
            morph = torch.tensor(morph)
        except:
            morph = torch.zeros(16, dtype=torch.float32)
            
        feat_path = os.path.join(self.features_dir, f"{img_name}.pt")
        try:
            feats = torch.load(feat_path, map_location='cpu', weights_only=True)
        except:
            feats = None
            
        label = self.label_map[row['label']]
        return img_tensor, spatial_3ch, morph, feats, label

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
    spatial_3chs = torch.stack([b[1] for b in batch])
    morphs = torch.stack([b[2] for b in batch])
    labels = torch.tensor([b[4] for b in batch])
    feats_list = [b[3] for b in batch]
    return img_tensors, spatial_3chs, morphs, feats_list, labels

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
    CLASSIFIER_PATH = r"artifacts/models/downstream/mlp_uni2h.joblib"
    
    timm_kwargs = {
        'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
        'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667*2,
        'num_classes': 0, 'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 'dynamic_img_size': True
    }
    
    # Load Frozen UNI-2h (Never mutated)
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
    
    train_dataset = UnifiedAblationDataset(train_df, morph_df, SPATIAL_DIR, DATA_DIR, transform)
    test_dataset = UnifiedAblationDataset(test_df, morph_df, SPATIAL_DIR, DATA_DIR, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, collate_fn=lambda x: x)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, collate_fn=lambda x: x)
    
    train_loader.collate_fn = collate_fn
    test_loader.collate_fn = collate_fn

    layers = [1, 12, 23]
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["qkv", "proj", "fc1", "fc2"],
        lora_dropout=0.1,
        bias="none",
    )
    
    # 1. Precompute Real Pathways (from the Test Set)
    print("Pre-computing True ViT Pathways on Test Set...")
    real_preds = []
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for img, _, _, _, _ in tqdm(test_loader, desc="Real Pathway"):
                img = img.to(device)
                true_emb = frozen_uni(img)
                probs = classifier.predict_proba(true_emb)
                preds = np.argmax(probs, axis=1)
                real_preds.extend(preds)
    real_preds = np.array(real_preds)
    
    results = []
    
    for layer in layers:
        print(f"\n{'='*50}\nTraining Adapter + LoRA for Layer {layer}\n{'='*50}")
        
        lora_uni = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        lora_uni = get_peft_model(lora_uni, lora_config)
        lora_uni.to(device)
        lora_uni.train()
        
        adapter = ResNetSpatialPatchPredictor().to(device)
        pred_path = os.path.join(MODEL_DIR, f"resnet_predictor_layer{layer}.pth")
        if os.path.exists(pred_path):
            adapter.load_state_dict(torch.load(pred_path, map_location=device))
        adapter.train()
        
        params = list(adapter.parameters()) + list(lora_uni.parameters())
        optimizer = optim.AdamW(params, lr=5e-5, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        epochs = 5
        
        for epoch in range(1, epochs + 1):
            adapter.train()
            lora_uni.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for _, spatial_3ch, morph, _, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
                spatial_3ch, morph, labels = spatial_3ch.to(device), morph.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    pred_seq = adapter(spatial_3ch, morph)
                    injected_emb = inject_forward(lora_uni, pred_seq, layer + 1)
                    logits = classifier.model(injected_emb)
                    loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * spatial_3ch.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            
        print(f"Training Complete for Layer {layer}. Evaluating against Frozen UNI-2h...")
        
        # 2. Evaluate Adapter against Frozen UNI-2h!
        adapter.eval()
        
        cossims = []
        frozen_injected_preds = []
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                for _, spatial_3ch, morph, feats_list, labels in tqdm(test_loader, desc=f"Eval Layer {layer}"):
                    spatial_3ch, morph = spatial_3ch.to(device), morph.to(device)
                    
                    # Get End-to-End trained Adapter output
                    pred_seq = adapter(spatial_3ch, morph)
                    
                    # Compute CosSim against True Frozen Embeddings
                    target_tensors = []
                    for f_dict in feats_list:
                        if f_dict is not None and layer in f_dict:
                            target_tensors.append(f_dict[layer])
                        else:
                            target_tensors.append(torch.zeros((265, 1536)))
                    target = torch.stack(target_tensors).to(device)
                    
                    pred_flat = pred_seq.flatten(1)
                    target_flat = target.flatten(1)
                    batch_cos = F.cosine_similarity(pred_flat, target_flat, dim=1).cpu().numpy()
                    cossims.extend(batch_cos)
                    
                    # Inject into FROZEN UNI-2H (Not LoRA!)
                    injected_emb = inject_forward(frozen_uni, pred_seq, layer + 1)
                    probs = classifier.predict_proba(injected_emb)
                    preds = np.argmax(probs, axis=1)
                    frozen_injected_preds.extend(preds)
                    
        frozen_injected_preds = np.array(frozen_injected_preds)
        avg_cossim = np.mean(cossims)
        agreement = np.mean(frozen_injected_preds == real_preds)
        
        print(f"Layer {layer} -> CosSim w/ Frozen: {avg_cossim:.4f} | Agreement w/ Frozen: {agreement*100:.2f}%")
        results.append({'Layer': layer, 'CosSim_to_Frozen': avg_cossim, 'Agreement_w_Frozen': agreement})
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(METRICS_DIR, "lora_ablation_audit_results.csv"), index=False)
    print("\nFinal Results:")
    print(res_df)

if __name__ == "__main__":
    main()
