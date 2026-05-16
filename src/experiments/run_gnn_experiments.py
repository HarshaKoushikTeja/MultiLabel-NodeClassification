"""
run_gnn_experiments.py — GNN Extension
Runs the full GNN benchmark: 3 models x 3 datasets x 5 seeds.
Saves results to results/gnn/gnn_results.csv and a combined comparison
table that includes the original DeepWalk/Node2Vec/baseline numbers.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gnn.trainer import run_multi_seed


DATASETS = ["blogcatalog", "ppi", "wikipedia"]
MODELS = ["gcn", "graphsage", "gat"]
SEEDS = (0, 1, 2, 3, 4)

# Per-dataset training config (tuned for reasonable performance)
CONFIG = {
    "blogcatalog": {"hidden_dim": 128, "lr": 0.01,  "max_epochs": 300, "patience": 40},
    "ppi":         {"hidden_dim": 128, "lr": 0.01,  "max_epochs": 300, "patience": 40},
    "wikipedia":   {"hidden_dim": 128, "lr": 0.01,  "max_epochs": 300, "patience": 40},
}

# Original results from DeepWalk/Node2Vec phase (from full_comparison.csv)
ORIGINAL_RESULTS = [
    {"dataset": "blogcatalog", "model": "Baseline",  "micro_f1_mean": 0.1652, "macro_f1_mean": 0.0245, "hamming_mean": 0.1375, "micro_f1_std": 0, "macro_f1_std": 0, "hamming_std": 0},
    {"dataset": "blogcatalog", "model": "DeepWalk",  "micro_f1_mean": 0.2851, "macro_f1_mean": 0.1443, "hamming_mean": 0.0317, "micro_f1_std": 0, "macro_f1_std": 0, "hamming_std": 0},
    {"dataset": "blogcatalog", "model": "Node2Vec",  "micro_f1_mean": 0.2941, "macro_f1_mean": 0.1476, "hamming_mean": 0.0313, "micro_f1_std": 0, "macro_f1_std": 0, "hamming_std": 0},
    {"dataset": "blogcatalog", "model": "Combined",  "micro_f1_mean": 0.3298, "macro_f1_mean": 0.1764, "hamming_mean": 0.0310, "micro_f1_std": 0, "macro_f1_std": 0, "hamming_std": 0},
    {"dataset": "ppi",         "model": "Baseline",  "micro_f1_mean": 0.0937, "macro_f1_mean": 0.0649, "hamming_mean": 0.4419, "micro_f1_std": 0, "macro_f1_std": 0, "hamming_std": 0},
    {"dataset": "ppi",         "model": "DeepWalk",  "micro_f1_mean": 0.0882, "macro_f1_mean": 0.0660, "hamming_mean": 0.0343, "micro_f1_std": 0, "macro_f1_std": 0, "hamming_std": 0},
    {"dataset": "ppi",         "model": "Node2Vec",  "micro_f1_mean": 0.0932, "macro_f1_mean": 0.0700, "hamming_mean": 0.0343, "micro_f1_std": 0, "macro_f1_std": 0, "hamming_std": 0},
    {"dataset": "ppi",         "model": "Combined",  "micro_f1_mean": 0.1184, "macro_f1_mean": 0.0944, "hamming_mean": 0.0348, "micro_f1_std": 0, "macro_f1_std": 0, "hamming_std": 0},
    {"dataset": "wikipedia",   "model": "Baseline",  "micro_f1_mean": 0.3859, "macro_f1_mean": 0.0292, "hamming_mean": 0.0524, "micro_f1_std": 0, "macro_f1_std": 0, "hamming_std": 0},
    {"dataset": "wikipedia",   "model": "DeepWalk",  "micro_f1_mean": 0.3423, "macro_f1_mean": 0.0415, "hamming_mean": 0.0326, "micro_f1_std": 0, "macro_f1_std": 0, "hamming_std": 0},
    {"dataset": "wikipedia",   "model": "Node2Vec",  "micro_f1_mean": 0.3479, "macro_f1_mean": 0.0475, "hamming_mean": 0.0323, "micro_f1_std": 0, "macro_f1_std": 0, "hamming_std": 0},
    {"dataset": "wikipedia",   "model": "Combined",  "micro_f1_mean": 0.3744, "macro_f1_mean": 0.0630, "hamming_mean": 0.0324, "micro_f1_std": 0, "macro_f1_std": 0, "hamming_std": 0},
]


def main():
    os.makedirs("results/gnn", exist_ok=True)
    all_results = []

    for dataset in DATASETS:
        cfg = CONFIG[dataset]
        for model in MODELS:
            result = run_multi_seed(dataset, model, seeds=SEEDS, **cfg)
            all_results.append(result)
            # Save incrementally so partial progress is never lost
            pd.DataFrame(all_results).to_csv(
                "results/gnn/gnn_results.csv", index=False
            )

    # Build combined comparison table (original + GNN)
    gnn_rows = []
    for r in all_results:
        gnn_rows.append({
            "dataset": r["dataset"],
            "model": r["model"].upper(),
            "micro_f1_mean": r["micro_f1_mean"], "micro_f1_std": r["micro_f1_std"],
            "macro_f1_mean": r["macro_f1_mean"], "macro_f1_std": r["macro_f1_std"],
            "hamming_mean":  r["hamming_mean"],  "hamming_std":  r["hamming_std"],
        })

    combined = pd.DataFrame(ORIGINAL_RESULTS + gnn_rows)
    combined.to_csv("results/gnn/full_comparison_with_gnn.csv", index=False)

    print(f"\n{'='*55}")
    print("ALL EXPERIMENTS COMPLETE")
    print('='*55)
    print("Saved:")
    print("  results/gnn/gnn_results.csv")
    print("  results/gnn/full_comparison_with_gnn.csv")
    print(f"\nTotal GNN experiments: {len(all_results)} "
          f"({len(MODELS)} models x {len(DATASETS)} datasets)")


if __name__ == "__main__":
    main()