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
    csv_frozen = os.path.join(METRICS_DIR, "resnet_forward_prop_audit_results.csv")
    csv_ablation = os.path.join(METRICS_DIR, "lora_ablation_audit_results.csv")
    
    if not os.path.exists(csv_frozen) or not os.path.exists(csv_ablation):
        print("Results CSV not found.")
        return
        
    df_frozen = pd.read_csv(csv_frozen)
    df_ablation = pd.read_csv(csv_ablation)
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid", context="talk")
    
    # Filter frozen down to layers 1, 12, 23 for fair comparison
    df_frozen_sub = df_frozen[df_frozen['Layer'].isin([1, 12, 23])]
    
    plt.plot(df_frozen_sub['Layer'], df_frozen_sub['Agreement'] * 100, marker='o', linewidth=3, markersize=10, 
             label='MSE-Trained Predictor -> Frozen ViT', color='#d62728', linestyle='--')
             
    plt.plot(df_ablation['Layer'], df_ablation['Agreement_w_Frozen'] * 100, marker='s', linewidth=3, markersize=10, 
             label='E2E LoRA-Trained Predictor -> Frozen ViT', color='#9467bd')
    
    plt.axhline(y=33.33, color='black', linestyle=':', linewidth=2, label='Random Chance (3-class)')
    
    plt.title('Ablation: Does End-to-End Training Un-Collapse the Frozen ViT?', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Injection Layer Depth (Block)', fontsize=14)
    plt.ylabel('Agreement w/ True Frozen Pathway (%)', fontsize=14)
    plt.ylim(20, 75)
    plt.xticks([1, 12, 23])
    
    plt.legend(loc='lower left', fontsize=12)
    plt.tight_layout()
    
    out_path = os.path.join(FIGURES_DIR, "lora_ablation_plot.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    main()
