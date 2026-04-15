"""
generate_baseline_results.py  —  Sai Sagar Galli Raghu | feature/evaluation
Reads existing prediction .npy files and ground truth labels,
computes Micro-F1, Macro-F1, and Hamming Loss, saves baseline_results.csv.

Run AFTER baseline.py has produced the prediction files.

Usage:
    python generate_baseline_results.py
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import f1_score, hamming_loss


# ── Config ────────────────────────────────────────────────────────────────────

DATASETS = {
    "blogcatalog": {
        "graph":       "/Users/adityakhurana/Developer/MultiLabel-NodeClassification/data/processed/blogcatalog.gpickle",
        "labels":      "/Users/adityakhurana/Developer/MultiLabel-NodeClassification/data/processed/labels_blogcatalog.csv",
        "test_idx":    "/Users/adityakhurana/Developer/MultiLabel-NodeClassification/data/processed/test_idx.npy",
        "pred":        "/Users/adityakhurana/Developer/MultiLabel-NodeClassification/src/classification/results/predictions_baseline_blogcatalog.npy",
        "idx_are_ids": True,
    },
    "ppi": {
        "graph":       "/Users/adityakhurana/Developer/MultiLabel-NodeClassification/data/processed/ppi.gpickle",
        "labels":      "/Users/adityakhurana/Developer/MultiLabel-NodeClassification/data/processed/labels_ppi.csv",
        "test_idx":    "/Users/adityakhurana/Developer/MultiLabel-NodeClassification/data/processed/test_idx_ppi.npy",
        "pred":        "/Users/adityakhurana/Developer/MultiLabel-NodeClassification/src/classification/results/predictions_baseline_ppi.npy",
        "idx_are_ids": False,
    },
    "wikipedia": {
        "graph":       "/Users/adityakhurana/Developer/MultiLabel-NodeClassification/data/processed/wikipedia.gpickle",
        "labels":      "/Users/adityakhurana/Developer/MultiLabel-NodeClassification/data/processed/labels_wikipedia.csv",
        "test_idx":    "/Users/adityakhurana/Developer/MultiLabel-NodeClassification/data/processed/test_idx_wikipedia.npy",
        "pred":        "/Users/adityakhurana/Developer/MultiLabel-NodeClassification/src/classification/results/predictions_baseline_wikipedia.npy",
        "idx_are_ids": False,
    },
}


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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rows = []

    for name, cfg in DATASETS.items():
        print(f"\nEvaluating: {name}")

        y_test = get_y_test(cfg)
        y_pred = np.load(cfg["pred"], allow_pickle=True)

        if y_pred.shape != y_test.shape:
            print(f"  WARNING: shape mismatch — y_pred {y_pred.shape} vs y_test {y_test.shape}")
            continue

        # Warn if predictions are all zeros (threshold too high)
        if y_pred.sum() == 0:
            print(f"  WARNING: all predictions are zero — re-run baseline.py to regenerate predictions with threshold tuning.")
            continue

        micro_f1  = f1_score(y_test, y_pred, average="micro", zero_division=0)
        macro_f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
        ham_loss  = hamming_loss(y_test, y_pred)

        print(f"  Micro-F1:     {micro_f1:.4f}")
        print(f"  Macro-F1:     {macro_f1:.4f}")
        print(f"  Hamming Loss: {ham_loss:.4f}")

        rows.append({
            "model":        "Baseline (degree + OvR LR)",
            "dataset":      name,
            "micro_f1":     round(micro_f1, 4),
            "macro_f1":     round(macro_f1, 4),
            "hamming_loss": round(ham_loss, 4),
        })

    if not rows:
        print("\nNo valid results to save. Check warnings above.")
        return

    os.makedirs("results/tables", exist_ok=True)
    out_path = "results/tables/baseline_results.csv"
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)

    print(f"\nSaved -> {out_path}")
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()