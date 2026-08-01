import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import joblib

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PyTorchMLPClassifier:
    def __init__(self, input_dim, hidden_layer_sizes=(512, 128), max_iter=100, lr=1e-3, device=DEVICE):
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

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)
        self.model = self._build_model(num_classes)
        
        y_mapped = np.array([np.where(self.classes_ == label)[0][0] for label in y])
        
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_mapped, dtype=torch.long).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(self.max_iter):
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        pred_idx = np.argmax(probs, axis=1)
        return self.classes_[pred_idx]

def main():
    parquet_path = r"data/processed/classification/tcga_subtypes/embeddings/combined/benchmark_features.parquet"
    manifest_dir = r"data/processed/classification/tcga_subtypes/manifests"
    model_dir = r"artifacts/models/downstream"
    os.makedirs(manifest_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print("Loading data...")
    df = pd.read_parquet(parquet_path)
    df = df[df['label'] != 'BRCA.Normal-like'].copy()
    df.reset_index(drop=True, inplace=True)
    
    X_uni = np.stack(df['uni2h'].values)
    X_ctr = np.stack(df['ctranspath'].values)
    X_res = np.stack(df['resnet50'].values)
    y = df['label'].values
    image_paths = df['image_path'].values
    
    from pathlib import Path
    # Parse patient ID from image path (e.g. TCGA-XX-XXXX)
    groups = np.array([str(Path(p).name).split('_')[0][:12] for p in image_paths])
    
    # We only care about finding high-confidence tiles for UNI-2h out-of-fold.
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(df), dtype=object)
    oof_probs = np.zeros(len(df))
    
    print("Running 5-fold CV to get out-of-fold predictions for UNI-2h...")
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X_uni, y, groups=groups)):
        X_train, X_test = X_uni[train_idx], X_uni[test_idx]
        y_train = y[train_idx]
        
        clf = PyTorchMLPClassifier(input_dim=X_uni.shape[1], hidden_layer_sizes=(512, 128))
        clf.fit(X_train, y_train)
        
        probs = clf.predict_proba(X_test)
        preds = clf.classes_[np.argmax(probs, axis=1)]
        conf = np.max(probs, axis=1)
        
        oof_preds[test_idx] = preds
        oof_probs[test_idx] = conf

    df['oof_pred'] = oof_preds
    df['oof_conf'] = oof_probs
    
    # Filter for hyper-confident and correct
    confident_correct = df[(df['label'] == df['oof_pred']) & (df['oof_conf'] > 0.8)].copy()
    print(f"Found {len(confident_correct)} correctly predicted, hyper-confident (>0.8) tiles.")
    
    # Sample up to 100
    n_sample = min(100, len(confident_correct))
    sampled_tiles = confident_correct.sample(n=n_sample, random_state=42)
    
    output_path = os.path.join(manifest_dir, "color_probe_samples.csv")
    sampled_tiles[['image_path', 'label', 'oof_pred', 'oof_conf']].to_csv(output_path, index=False)
    print(f"Saved {n_sample} target tiles to {output_path}")
    
    # Train final models on FULL dataset
    print("Training final PyTorch MLPs on full 1500-sample dataset...")
    
    clf_uni = PyTorchMLPClassifier(input_dim=X_uni.shape[1])
    clf_uni.fit(X_uni, y)
    joblib.dump(clf_uni, os.path.join(model_dir, "mlp_uni2h.joblib"))
    print("Saved UNI2-h MLP")
    
    clf_ctr = PyTorchMLPClassifier(input_dim=X_ctr.shape[1])
    clf_ctr.fit(X_ctr, y)
    joblib.dump(clf_ctr, os.path.join(model_dir, "mlp_ctranspath.joblib"))
    print("Saved CTransPath MLP")
    
    clf_res = PyTorchMLPClassifier(input_dim=X_res.shape[1])
    clf_res.fit(X_res, y)
    joblib.dump(clf_res, os.path.join(model_dir, "mlp_resnet50.joblib"))
    print("Saved ResNet50 MLP")
    
    print("Done!")

if __name__ == "__main__":
    main()
