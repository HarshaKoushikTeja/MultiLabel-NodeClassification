"""
plot_comparison.py — GNN Extension
The headline comparison chart: all 7 methods (Baseline, DeepWalk, Node2Vec,
Combined, GCN, GraphSAGE, GAT) across 3 datasets, with error bars for GNNs.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("results/gnn/full_comparison_with_gnn.csv")

DATASETS = ["blogcatalog", "ppi", "wikipedia"]
METHOD_ORDER = ["Baseline", "DeepWalk", "Node2Vec", "Combined",
                "GCN", "GRAPHSAGE", "GAT"]
COLORS = {
    "Baseline": "#94A3B8", "DeepWalk": "#1C7293", "Node2Vec": "#1A8A63",
    "Combined": "#CA8A04", "GCN": "#7C3AED", "GRAPHSAGE": "#DC2626",
    "GAT": "#F97316",
}

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for ax, dataset in zip(axes, DATASETS):
    sub = df[df["dataset"] == dataset].set_index("model")
    methods = [m for m in METHOD_ORDER if m in sub.index]
    means = [sub.loc[m, "micro_f1_mean"] for m in methods]
    stds  = [sub.loc[m, "micro_f1_std"]  for m in methods]
    colors = [COLORS[m] for m in methods]

    bars = ax.bar(methods, means, yerr=stds, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.6,
                  error_kw={"elinewidth": 1.2})
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold")

    ax.set_title(dataset.upper(), fontsize=14, fontweight="bold")
    ax.set_ylabel("Micro-F1" if dataset == "blogcatalog" else "")
    ax.set_ylim(0, max(means) * 1.25)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("Classical Embeddings vs Modern GNNs — Micro-F1 across 3 datasets\n"
             "(GNN bars show mean ± std over 5 seeds)",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
os.makedirs("results/visualizations", exist_ok=True)
out = "results/visualizations/classical_vs_gnn_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved -> {out}")