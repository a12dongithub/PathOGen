import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    DATA_DIR = r"data/misc/tcga_10k_cached_tensors"
    META_PATH = r"data/processed/classification/tcga_subtypes/manifests/legacy_10k_samples.csv"
    MODEL_DIR = r"artifacts/runs/legacy_representation_audit/models"
    METRICS_DIR = r"artifacts/runs/legacy_representation_audit/metrics"
    FIGURES_DIR = r"artifacts/runs/legacy_representation_audit/figures"
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    csv_path = os.path.join(METRICS_DIR, "resnet_forward_prop_audit_results.csv")
    
    if not os.path.exists(csv_path):
        print("Results CSV not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid", context="talk")
    
    plt.plot(df['Layer'], df['Agreement'] * 100, marker='o', linewidth=3, markersize=10, 
             label='Agreement w/ Real Pathway', color='#2ca02c')
    plt.plot(df['Layer'], df['Accuracy'] * 100, marker='s', linewidth=3, markersize=10, 
             label='Downstream Accuracy', color='#1f77b4', linestyle='--')
    
    plt.axhline(y=33.33, color='r', linestyle=':', linewidth=2, label='Random Chance (3-class)')
    
    plt.title('ResNet Synthetic Spatial-Token Injection\n(Predicting UNI-2h Intermediate Layers)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Injection Layer Depth (Block)', fontsize=14)
    plt.ylabel('Performance (%)', fontsize=14)
    plt.ylim(20, 60)
    plt.xticks([1, 3, 6, 12, 23])
    
    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    
    out_path = os.path.join(FIGURES_DIR, "resnet_forward_prop_audit_plot.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    main()
