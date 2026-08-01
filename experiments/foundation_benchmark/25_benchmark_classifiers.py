import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import time

class PyTorchMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class PyTorchMLPClassifier:
    def __init__(self, max_iter=100, batch_size=256, lr=1e-3, device='cuda'):
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)
        input_dim = X.shape[1]
        
        self.model = PyTorchMLP(input_dim, num_classes).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.max_iter):
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                out = self.model(batch_X)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            tensor_X = torch.tensor(X, dtype=torch.float32).to(self.device)
            out = self.model(tensor_X)
            preds = torch.argmax(out, dim=1).cpu().numpy()
        return preds

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            tensor_X = torch.tensor(X, dtype=torch.float32).to(self.device)
            out = self.model(tensor_X)
            probs = torch.softmax(out, dim=1).cpu().numpy()
        return probs
    
    def get_params(self, deep=True):
        return {"max_iter": self.max_iter, "batch_size": self.batch_size, "lr": self.lr, "device": self.device.type if hasattr(self.device, 'type') else self.device}

def main():
    print("Loading extracted features...")
    parquet_path = r"data/processed/classification/tcga_subtypes/embeddings/combined/benchmark_features.parquet"
    if not os.path.exists(parquet_path):
        print(f"File not found: {parquet_path}")
        return
        
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} samples originally.")
    
    # Filter out Normal-like class
    df = df[df['label'] != 'BRCA.Normal-like'].copy()
    print(f"Loaded {len(df)} samples after removing Normal-like.")
    
    # Extract features
    X_uni = np.stack(df['uni2h'].values)
    X_ctp = np.stack(df['ctranspath'].values)
    X_res = np.stack(df['resnet50'].values) if 'resnet50' in df.columns else None
    
    labels_str = df['label'].values
    unique_classes = sorted(list(set(labels_str)))
    print(f"Classes: {unique_classes}")
    
    le = LabelEncoder()
    y = le.fit_transform(labels_str)

    # Extract Patient IDs to prevent data leakage across folds
    patient_ids = []
    for path in df['image_path']:
        basename = os.path.basename(path)
        pid = "-".join(basename.split("_")[0].split("-")[:3])
        patient_ids.append(pid)
    groups = np.array(patient_ids)
    print(f"Found {len(set(groups))} unique patients.")

    # Define classifiers
    models_dict = {
        "LR": LogisticRegression(max_iter=1000, random_state=42),
        "RF": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGB": XGBClassifier(device='cuda', tree_method='hist', use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        "MLP (PyTorch)": PyTorchMLPClassifier(max_iter=100, batch_size=64, lr=1e-3)
    }
    
    features_dict = {
        "UNI-2h": X_uni,
        "CTransPath": X_ctp
    }
    if X_res is not None:
        features_dict["ResNet50"] = X_res
    
    results = []
    
    # Grouped Split
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    for feat_name, X in features_dict.items():
        print(f"\n==================================================")
        print(f"Evaluating Features: {feat_name} (shape: {X.shape})")
        
        for clf_name, clf in models_dict.items():
            print(f"  Training {clf_name}...")
            
            fold_accs = []
            fold_accs_patient_mean = []
            fold_accs_patient_max = []
            fold_aucs_macro = []
            fold_aucs_overall = []
            
            start_time = time.time()
            
            for train_idx, test_idx in sgkf.split(X, y, groups=groups):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                groups_test = groups[test_idx]
                
                clf_clone = type(clf)(**clf.get_params())
                clf_clone.fit(X_train, y_train)
                
                y_pred = clf_clone.predict(X_test)
                fold_accs.append(accuracy_score(y_test, y_pred))
                
                if hasattr(clf_clone, "predict_proba"):
                    y_prob = clf_clone.predict_proba(X_test)
                    
                    try:
                        macro_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro', labels=np.unique(y))
                        fold_aucs_macro.append(macro_auc)
                    except Exception as e:
                        print(f"Warning: ROC AUC failed for this fold: {e}")
                    
                    try:
                        overall_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='micro', labels=np.unique(y))
                        fold_aucs_overall.append(overall_auc)
                    except Exception as e:
                        pass
                    
                    # --- Patient-Level Aggregation ---
                    unique_patients = np.unique(groups_test)
                    patient_y_true = []
                    patient_y_pred_mean = []
                    patient_y_pred_max = []
                    
                    for pat in unique_patients:
                        idx = np.where(groups_test == pat)[0]
                        pat_true = y_test[idx][0]  # All tiles for a patient have the same label
                        pat_probs = y_prob[idx]
                        
                        # Mean aggregation
                        mean_prob = np.mean(pat_probs, axis=0)
                        # Max aggregation
                        max_prob = np.max(pat_probs, axis=0)
                        
                        patient_y_true.append(pat_true)
                        patient_y_pred_mean.append(np.argmax(mean_prob))
                        patient_y_pred_max.append(np.argmax(max_prob))
                        
                    fold_accs_patient_mean.append(accuracy_score(patient_y_true, patient_y_pred_mean))
                    fold_accs_patient_max.append(accuracy_score(patient_y_true, patient_y_pred_max))
                    
            end_time = time.time()
            
            mean_acc = np.mean(fold_accs)
            mean_acc_patient_mean = np.mean(fold_accs_patient_mean) if fold_accs_patient_mean else np.nan
            mean_acc_patient_max = np.mean(fold_accs_patient_max) if fold_accs_patient_max else np.nan
            mean_auc_macro = np.mean(fold_aucs_macro) if fold_aucs_macro else np.nan
            mean_auc_overall = np.mean(fold_aucs_overall) if fold_aucs_overall else np.nan
            
            print(f"    -> Acc (Tile): {mean_acc*100:.2f}% | Acc (Patient-Mean): {mean_acc_patient_mean*100:.2f}% | Acc (Patient-Max): {mean_acc_patient_max*100:.2f}% | AUC (Macro): {mean_auc_macro:.4f} | Time: {end_time-start_time:.1f}s")
            
            results.append({
                "Feature": feat_name,
                "Classifier": clf_name,
                "Accuracy (Tile) %": round(mean_acc * 100, 2),
                "Accuracy (Patient-Mean) %": round(mean_acc_patient_mean * 100, 2) if not np.isnan(mean_acc_patient_mean) else None,
                "Accuracy (Patient-Max) %": round(mean_acc_patient_max * 100, 2) if not np.isnan(mean_acc_patient_max) else None,
                "AUC (Macro)": round(mean_auc_macro, 4),
                "AUC (Overall)": round(mean_auc_overall, 4)
            })
            
    # Save to CSV
    df_results = pd.DataFrame(results)
    out_csv = r"artifacts/runs/legacy_foundation_benchmark/metrics/benchmark_metrics.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_results.to_csv(out_csv, index=False)
    print(f"\nSaved metrics to {out_csv}")
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    main()
