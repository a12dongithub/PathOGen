import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score
from scipy.stats import pearsonr
import joblib

# PyTorchMLPClassifier definition required for unpickling
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
        pass # Not needed for evaluation

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

# Import predictor from previous script
from importlib.machinery import SourceFileLoader
mod = SourceFileLoader("predictor", r"experiments/representation_audit/31_train_activation_predictor.py").load_module()
ActivationPredictor = mod.ActivationPredictor
ActivationDataset = mod.ActivationDataset

BENCHMARK_FEATURES_PATH = r"data/processed/classification/tcga_subtypes/embeddings/combined/benchmark_features.parquet"
MORPH_STATS_PATH = r"data/processed/conditions/morphology/standardized.parquet"
SPATIAL_DIR = r"data/processed/conditions/spatial_maps"
OUTPUT_DIR = r"artifacts/runs/activation_poc"

MLP_UNI_PATH = r"artifacts/models/downstream/mlp_uni2h.joblib"

def evaluate_downstream():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load trained Activation Predictor
    model_path = os.path.join(OUTPUT_DIR, "best_activation_predictor.pth")
    model = ActivationPredictor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # 2. Load Downstream Classifier (MLP trained on real UNI-2h features)
    classifier = joblib.load(MLP_UNI_PATH)
    
    # 3. Load Test Data
    test_meta_path = os.path.join(OUTPUT_DIR, "test_set_metadata.csv")
    test_df_meta = pd.read_csv(test_meta_path)
    
    # Need true features
    all_features = pd.read_parquet(BENCHMARK_FEATURES_PATH)
    test_df = all_features[all_features['image_path'].isin(test_df_meta['image_path'])].copy()
    
    morph_df = pd.read_parquet(MORPH_STATS_PATH)
    test_dataset = ActivationDataset(test_df, morph_df, SPATIAL_DIR)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 4. Generate Predictions
    true_acts = []
    pred_acts = []
    labels = test_df['label'].values
    
    with torch.no_grad():
        for spatial, morph, target, _ in test_loader:
            spatial, morph = spatial.to(device), morph.to(device)
            pred = model(spatial, morph)
            
            true_acts.append(target.cpu().numpy())
            pred_acts.append(pred.cpu().numpy())
            
    true_acts = np.concatenate(true_acts, axis=0)
    pred_acts = np.concatenate(pred_acts, axis=0)
    
    # Calculate Activation Reconstruction Metrics
    mse = np.mean((true_acts - pred_acts) ** 2)
    
    # Cosine Similarity
    true_norm = true_acts / np.linalg.norm(true_acts, axis=1, keepdims=True)
    pred_norm = pred_acts / np.linalg.norm(pred_acts, axis=1, keepdims=True)
    cos_sims = np.sum(true_norm * pred_norm, axis=1)
    mean_cos_sim = np.mean(cos_sims)
    
    # Pearson R (average across all samples)
    pearsons = [pearsonr(true_acts[i], pred_acts[i])[0] for i in range(len(true_acts))]
    mean_pearson = np.mean(pearsons)
    
    # 5. Downstream Classifier Evaluation
    # Note: labels are strings (e.g., 'BRCA.Basal', 'BRCA.LumA'). 
    # The joblib MLP predict_proba expects numeric features.
    
    true_probs = classifier.predict_proba(true_acts)
    pred_probs = classifier.predict_proba(pred_acts)
    
    true_preds = classifier.classes_[np.argmax(true_probs, axis=1)]
    pred_preds = classifier.classes_[np.argmax(pred_probs, axis=1)]
    
    # Metrics
    agreement = accuracy_score(true_preds, pred_preds)
    
    acc_true = accuracy_score(labels, true_preds)
    acc_pred = accuracy_score(labels, pred_preds)
    
    # Multi-class AUC
    # We must binarize labels for sklearn roc_auc_score or let it handle it via ovo/ovr.
    # We will use 'ovr' (One-vs-Rest)
    try:
        auc_true = roc_auc_score(labels, true_probs, multi_class='ovr')
        auc_pred = roc_auc_score(labels, pred_probs, multi_class='ovr')
    except Exception as e:
        auc_true = -1
        auc_pred = -1
        print("AUC calc failed:", e)

    print("==========================================")
    print(" ACTIVATION RECONSTRUCTION METRICS")
    print("==========================================")
    print(f"Mean Cosine Similarity : {mean_cos_sim:.4f}")
    print(f"Mean Squared Error (MSE) : {mse:.4f}")
    print(f"Mean Pearson Correlation : {mean_pearson:.4f}")
    print()
    print("==========================================")
    print(" DOWNSTREAM CLASSIFIER PRESERVATION")
    print("==========================================")
    print(f"Top-1 Prediction Agreement : {agreement*100:.2f}%")
    print(f"TRUE Activation Accuracy : {acc_true*100:.2f}%")
    print(f"PRED Activation Accuracy : {acc_pred*100:.2f}%")
    print(f"TRUE Activation AUC : {auc_true:.4f}")
    print(f"PRED Activation AUC : {auc_pred:.4f}")
    print("==========================================")

if __name__ == "__main__":
    evaluate_downstream()
