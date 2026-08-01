import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_batch_results():
    OUTPUT_DIR = Path(r"artifacts/runs/batch_probe/results")
    csv_file = OUTPUT_DIR / "batch_probe_results.csv"
    
    if not csv_file.exists():
        print(f"Error: {csv_file} does not exist.")
        return
        
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows.")
    
    classes = ["Normal", "Benign", "InSitu", "Invasive"]
    interventions = ["Original", "All Tumor", "All Immune", "All Stroma"]
    
    # Ensure correct order of interventions for plotting
    df['Intervention'] = pd.Categorical(df['Intervention'], categories=interventions, ordered=True)
    
    # 1. Plot Average Probabilities (Grouped Bar Chart)
    avg_probs = df.groupby("Intervention")[[f"Prob_{c}" for c in classes]].mean().reset_index()
    
    # Melt for seaborn
    melted_avg = avg_probs.melt(id_vars="Intervention", var_name="Class", value_name="Probability")
    melted_avg["Class"] = melted_avg["Class"].str.replace("Prob_", "")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted_avg, x="Intervention", y="Probability", hue="Class", palette="viridis")
    plt.title("Average Class Probabilities Across Interventions (N=50)")
    plt.ylabel("Mean Probability")
    plt.xlabel("Spatial Intervention")
    plt.ylim(0, 1)
    plt.legend(title="Predicted Subtype")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "avg_probabilities_bar.png", dpi=300)
    plt.close()
    print("Saved avg_probabilities_bar.png")
    
    # 2. Plot Predicted Class Distribution (Stacked Bar Chart)
    # Count how many of each class were predicted per intervention
    class_counts = df.groupby(["Intervention", "Predicted_Class"]).size().unstack(fill_value=0)
    
    # Ensure all classes are present in columns
    for c in classes:
        if c not in class_counts.columns:
            class_counts[c] = 0
            
    class_counts = class_counts[classes] # reorder columns
    
    # Convert to percentages
    class_percentages = class_counts.div(class_counts.sum(axis=1), axis=0) * 100
    
    ax = class_percentages.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="viridis")
    plt.title("Distribution of Predicted Subtypes Across Interventions (N=50)")
    plt.ylabel("Percentage of Tiles (%)")
    plt.xlabel("Spatial Intervention")
    
    # Move legend outside
    plt.legend(title="Predicted Subtype", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predicted_class_distribution_stacked.png", dpi=300)
    plt.close()
    print("Saved predicted_class_distribution_stacked.png")

if __name__ == "__main__":
    plot_batch_results()
