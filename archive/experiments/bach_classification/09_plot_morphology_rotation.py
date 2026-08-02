import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def plot_morphology_rotation():
    OUTPUT_DIR = Path(r"artifacts/runs/morphology_rotation")
    csv_file = OUTPUT_DIR / "morphology_rotation_results.csv"
    
    if not csv_file.exists():
        print(f"Error: {csv_file} does not exist.")
        return
        
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows.")
    
    classes = ["Normal", "Benign", "InSitu", "Invasive"]
    
    # ── 1. Rotation Invariance Plot ──
    rot_interventions = ["Original", "Rot 90", "Rot 180", "Rot 270"]
    df_rot = df[df['Intervention'].isin(rot_interventions)].copy()
    df_rot['Intervention'] = pd.Categorical(df_rot['Intervention'], categories=rot_interventions, ordered=True)
    
    avg_probs_rot = df_rot.groupby("Intervention")[[f"Prob_{c}" for c in classes]].mean().reset_index()
    melted_rot = avg_probs_rot.melt(id_vars="Intervention", var_name="Class", value_name="Probability")
    melted_rot["Class"] = melted_rot["Class"].str.replace("Prob_", "")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted_rot, x="Intervention", y="Probability", hue="Class", palette="viridis")
    plt.title("Rotation Invariance of UNI2-h on Generated Tiles (N=10)")
    plt.ylabel("Mean Probability")
    plt.xlabel("Spatial Map Rotation")
    plt.ylim(0, 1)
    plt.legend(title="Predicted Subtype")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rotation_invariance_bar.png", dpi=300)
    plt.close()
    print("Saved rotation_invariance_bar.png")
    
    # ── 2. Swap Interventions Plot ──
    morph_interventions = ["Original", "Morph Swap", "Spatial Swap"]
    df_morph = df[df['Intervention'].isin(morph_interventions)].copy()
    df_morph['Intervention'] = pd.Categorical(df_morph['Intervention'], categories=morph_interventions, ordered=True)
    
    avg_probs_morph = df_morph.groupby("Intervention")[[f"Prob_{c}" for c in classes]].mean().reset_index()
    melted_morph = avg_probs_morph.melt(id_vars="Intervention", var_name="Class", value_name="Probability")
    melted_morph["Class"] = melted_morph["Class"].str.replace("Prob_", "")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted_morph, x="Intervention", y="Probability", hue="Class", palette="plasma")
    plt.title("Impact of Morphology and Spatial Swaps on UNI2-h Predictions (N=10)")
    plt.ylabel("Mean Probability")
    plt.xlabel("Intervention")
    plt.ylim(0, 1)
    plt.legend(title="Predicted Subtype")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "morphology_swap_bar.png", dpi=300)
    plt.close()
    print("Saved morphology_swap_bar.png")
    
    # ── 3. Calculate Mean Absolute Probability Shift ──
    # How much did the probabilities shift on average per sample when we swapped morphology?
    orig_df = df[df['Intervention'] == 'Original'].set_index('Sample_ID')
    swap_df = df[df['Intervention'] == 'Morph Swap'].set_index('Sample_ID')
    
    diffs = []
    for c in classes:
        col = f"Prob_{c}"
        mae = np.mean(np.abs(orig_df[col] - swap_df[col]))
        diffs.append((c, mae))
        
    print("\nMean Absolute Probability Shift caused by Morphology Swap:")
    for c, mae in diffs:
        print(f"  {c}: {mae*100:.2f}%")
        
    # Same for Spatial Swap
    spatial_swap_df = df[df['Intervention'] == 'Spatial Swap'].set_index('Sample_ID')
    spatial_diffs = []
    for c in classes:
        col = f"Prob_{c}"
        mae = np.mean(np.abs(orig_df[col] - spatial_swap_df[col]))
        spatial_diffs.append((c, mae))
        
    print("\nMean Absolute Probability Shift caused by Spatial Swap:")
    for c, mae in spatial_diffs:
        print(f"  {c}: {mae*100:.2f}%")
        
    # Same for Rot 90
    rot90_df = df[df['Intervention'] == 'Rot 90'].set_index('Sample_ID')
    rot_diffs = []
    for c in classes:
        col = f"Prob_{c}"
        mae = np.mean(np.abs(orig_df[col] - rot90_df[col]))
        rot_diffs.append((c, mae))
        
    print("\nMean Absolute Probability Shift caused by 90-degree Rotation:")
    for c, mae in rot_diffs:
        print(f"  {c}: {mae*100:.2f}%")

if __name__ == "__main__":
    plot_morphology_rotation()
