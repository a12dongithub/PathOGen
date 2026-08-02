import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    results = {
        2: 52.97,
        4: 39.03,
        6: 55.32,
        8: 45.03,
        10: 43.88,
        12: 44.48,
        14: 48.43,
        16: 53.32,
        18: 55.02,
        20: 54.42,
        22: 54.02,
        24: 55.92
    }
    
    layers = sorted(results.keys())
    accuracies = [results[l] for l in layers]
    
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    # Plotting line
    sns.lineplot(x=layers, y=accuracies, marker='o', linewidth=2, markersize=8, color='crimson')
    
    # Annotate points
    for x, y in zip(layers, accuracies):
        plt.text(x, y + 0.5, f"{y:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.title("Adapter Injection Accuracy Across UNI-2h Layers (Even Sweep)", fontsize=14, fontweight='bold')
    plt.xlabel("Injection Layer", fontsize=12)
    plt.ylabel("Prediction Agreement (%)", fontsize=12)
    plt.xticks(layers)
    plt.ylim(30, 65)
    
    plt.tight_layout()
    output = Path("artifacts/runs/legacy_representation_audit/figures/even_layer_sweep_results.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300)
    print(f"Saved plot to {output}")

if __name__ == "__main__":
    main()
