"""
generate_full_results.py  —  Sai Sagar Galli Raghu | feature/evaluation
Reads existing prediction .npy files for all models and datasets,
computes Micro-F1, Macro-F1, and Hamming Loss, saves full_comparison.csv.

Run AFTER classifier.py and baseline.py have produced all prediction files.

Usage:
    python generate_full_results.py

Inputs (must already exist in results/):
    predictions_baseline_blogcatalog.npy
    predictions_baseline_ppi.npy
    predictions_baseline_wikipedia.npy
    predictions_deepwalk_blogcatalog.npy
    predictions_deepwalk_ppi.npy
    predictions_deepwalk_wikipedia.npy
    predictions_node2vec_blogcatalog.npy
    predictions_node2vec_ppi.npy
    predictions_node2vec_wikipedia.npy
    predictions_combined_blogcatalog.npy
    predictions_combined_ppi.npy
    predictions_combined_wikipedia.npy

Output:
    results/tables/full_comparison.csv
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import f1_score, hamming_loss


# ── Config ────────────────────────────────────────────────────────────────────

DATASETS = {
    "blogcatalog": {
        "graph":       "data/processed/blogcatalog.gpickle",
        "labels":      "data/processed/labels_blogcatalog.csv",
        "test_idx":    "data/processed/test_idx.npy",
        "idx_are_ids": True,
    },
    "ppi": {
        "graph":       "data/processed/ppi.gpickle",
        "labels":      "data/processed/labels_ppi.csv",
        "test_idx":    "data/processed/test_idx_ppi.npy",
        "idx_are_ids": False,
    },
    "wikipedia": {
        "graph":       "data/processed/wikipedia.gpickle",
        "labels":      "data/processed/labels_wikipedia.csv",
        "test_idx":    "data/processed/test_idx_wikipedia.npy",
        "idx_are_ids": False,
    },
}

# All models to evaluate — (display name, prediction file prefix)
MODELS = [
    ("Baseline (degree + OvR LR)", "baseline"),
    ("DeepWalk",                   "deepwalk"),
    ("Node2Vec",                   "node2vec"),
    ("Combined (DW+N2V)",          "combined"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_y_test(cfg: dict) -> np.ndarray:
    """Reconstruct ground truth y_test from labels CSV + test indices."""
    with open(cfg["graph"], "rb") as f:
        G = pickle.load(f)

    nodes = sorted(G.nodes())
    node_to_pos = {n: i for i, n in enumerate(nodes)}

    df = pd.read_csv(cfg["labels"]).set_index("node_id")
    label_cols = df.columns.tolist()

    y = np.zeros((len(nodes), len(label_cols)), dtype=int)
    for node in nodes:
        if node in df.index:
            y[node_to_pos[node]] = df.loc[node, label_cols].values.astype(int)

    raw_test = np.load(cfg["test_idx"], allow_pickle=True)
    if cfg["idx_are_ids"]:
        test_pos = np.array([node_to_pos[nid] for nid in raw_test])
    else:
        test_pos = raw_test

    return y[test_pos]


def compute_metrics(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute Micro-F1, Macro-F1, and Hamming Loss."""
    return {
        "micro_f1":     round(f1_score(y_test, y_pred, average="micro", zero_division=0), 4),
        "macro_f1":     round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "hamming_loss": round(hamming_loss(y_test, y_pred), 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rows = []

    for dataset, cfg in DATASETS.items():
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'='*50}")

        y_test = get_y_test(cfg)

        for display_name, prefix in MODELS:
            pred_path = f"results/predictions_{prefix}_{dataset}.npy"

            # Skip cleanly if prediction file doesn't exist yet
            if not os.path.exists(pred_path):
                print(f"  {display_name:<25} SKIPPED — {pred_path} not found")
                continue

            y_pred = np.load(pred_path, allow_pickle=True)

            # Shape check
            if y_pred.shape != y_test.shape:
                print(f"  {display_name:<25} WARNING — shape mismatch: "
                      f"y_pred {y_pred.shape} vs y_test {y_test.shape}")
                continue

            # All-zero check
            if y_pred.sum() == 0:
                print(f"  {display_name:<25} WARNING — all predictions are zero. "
                      f"Re-run baseline.py with threshold tuning.")
                continue

            m = compute_metrics(y_test, y_pred)
            print(f"  {display_name:<25} "
                  f"micro={m['micro_f1']:.4f}  "
                  f"macro={m['macro_f1']:.4f}  "
                  f"hamming={m['hamming_loss']:.4f}")

            rows.append({
                "model":        display_name,
                "dataset":      dataset,
                "micro_f1":     m["micro_f1"],
                "macro_f1":     m["macro_f1"],
                "hamming_loss": m["hamming_loss"],
            })

    if not rows:
        print("\nNo valid results to save. Check warnings above.")
        return

    os.makedirs("results/tables", exist_ok=True)
    out_path = "results/tables/full_comparison.csv"
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)

    print(f"\nSaved -> {out_path}")
    print(f"\n{df_out.to_string(index=False)}")


if __name__ == "__main__":
    main()
