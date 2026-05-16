"""
visualize_tsne.py — GNN Extension
Creates t-SNE projections of learned embeddings, colored by dominant label.
This is the "killer plot" — shows visually how well each method separates classes.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gnn.data_loader import build_pyg_data


def dominant_label(y):
    """For multi-label, pick the first active label as the color (for viz only)."""
    labels = np.full(y.shape[0], -1)
    for i in range(y.shape[0]):
        active = np.where(y[i] > 0)[0]
        if len(active) > 0:
            labels[i] = active[0]
    return labels


def plot_embedding_tsne(emb, y, title, out_path, max_points=2000):
    """Project embeddings to 2D with t-SNE and scatter-plot by dominant label."""
    labels = dominant_label(y)
    mask = labels >= 0
    emb, labels = emb[mask], labels[mask]

    # Subsample for speed if large
    if emb.shape[0] > max_points:
        idx = np.random.RandomState(42).choice(emb.shape[0], max_points, replace=False)
        emb, labels = emb[idx], labels[idx]

    print(f"  Running t-SNE on {emb.shape[0]} points (dim {emb.shape[1]})...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                init="pca", learning_rate="auto")
    emb_2d = tsne.fit_transform(emb)

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels,
                          cmap="tab20", s=12, alpha=0.7)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.colorbar(scatter, label="Dominant label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path}")


def main():
    os.makedirs("results/visualizations", exist_ok=True)

    # Visualize the existing DeepWalk and Node2Vec embeddings
    for dataset in ["blogcatalog", "ppi", "wikipedia"]:
        data = build_pyg_data(dataset)
        y = data.y.cpu().numpy()

        for method in ["deepwalk", "node2vec"]:
            emb_path = f"results/{method}_{dataset}.npy"
            if not os.path.exists(emb_path):
                print(f"  Skip: {emb_path} not found")
                continue
            emb = np.load(emb_path)
            title = f"{method.upper()} embeddings — {dataset.upper()}"
            out = f"results/visualizations/tsne_{method}_{dataset}.png"
            print(f"\n{title}")
            plot_embedding_tsne(emb, y, title, out)

    print("\nAll t-SNE visualizations saved to results/visualizations/")

def plot_comparison_grid():
    """Create a 3x2 grid: rows=datasets, cols=[DeepWalk, Node2Vec]."""
    datasets = ["blogcatalog", "ppi", "wikipedia"]
    methods = ["deepwalk", "node2vec"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 18))

    for row, dataset in enumerate(datasets):
        data = build_pyg_data(dataset)
        y = data.y.cpu().numpy()
        labels_full = dominant_label(y)

        for col, method in enumerate(methods):
            ax = axes[row, col]
            emb_path = f"results/{method}_{dataset}.npy"
            if not os.path.exists(emb_path):
                ax.set_title(f"{method} {dataset} (missing)")
                continue

            emb = np.load(emb_path)
            labels = labels_full.copy()
            mask = labels >= 0
            emb_m, labels_m = emb[mask], labels[mask]

            if emb_m.shape[0] > 2000:
                idx = np.random.RandomState(42).choice(
                    emb_m.shape[0], 2000, replace=False)
                emb_m, labels_m = emb_m[idx], labels_m[idx]

            print(f"  t-SNE: {method} {dataset} ({emb_m.shape[0]} pts)")
            tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                        init="pca", learning_rate="auto")
            emb_2d = tsne.fit_transform(emb_m)

            sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels_m,
                            cmap="tab20", s=10, alpha=0.6)
            ax.set_title(f"{method.upper()} — {dataset.upper()}",
                         fontsize=13, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Graph Embedding Visualizations (t-SNE)\n"
                 "DeepWalk vs Node2Vec across 3 datasets",
                 fontsize=16, fontweight="bold", y=1.0)
    plt.tight_layout()
    out = "results/visualizations/tsne_comparison_grid.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved comparison grid -> {out}")


if __name__ == "__main__":
    os.makedirs("results/visualizations", exist_ok=True)
    plot_comparison_grid()