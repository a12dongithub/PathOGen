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
    csv_lora = os.path.join(METRICS_DIR, "lora_aligned_audit_results.csv")
    
    if not os.path.exists(csv_frozen) or not os.path.exists(csv_lora):
        print("Results CSV not found.")
        return
        
    df_frozen = pd.read_csv(csv_frozen)
    df_lora = pd.read_csv(csv_lora)
    
    # Merge on Layer
    df = pd.merge(df_frozen, df_lora, on='Layer')
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid", context="talk")
    
    plt.plot(df['Layer'], df['Accuracy'] * 100, marker='o', linewidth=3, markersize=10, 
             label='Frozen UNI-2h (Strict Audit)', color='#d62728', linestyle='--')
             
    plt.plot(df['Layer'], df['Aligned_Accuracy'] * 100, marker='s', linewidth=3, markersize=10, 
             label='LoRA Hybrid (Spatial-ViT Re-alignment)', color='#2ca02c')
    
    plt.axhline(y=33.33, color='black', linestyle=':', linewidth=2, label='Random Chance (3-class)')
    
    plt.title('Hybrid Model: LoRA Re-alignment vs Frozen ViT\n(Accuracy given Spatial/Morph Inputs)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Injection Layer Depth (Block)', fontsize=14)
    plt.ylabel('Downstream Accuracy (%)', fontsize=14)
    plt.ylim(20, 75)
    plt.xticks([1, 3, 6, 12, 23])
    
    plt.legend(loc='lower left', fontsize=12)
    plt.tight_layout()
    
    out_path = os.path.join(FIGURES_DIR, "lora_alignment_plot.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    main()
