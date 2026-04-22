"""
run_deepwalk.py  —  Shaman Kanapathy | feature/deepwalk
Runs DeepWalk on all three datasets and saves embedding .npy files.

Usage:
    python run_deepwalk.py

Outputs:
    results/deepwalk_blogcatalog.npy   shape: (10312, 128)
    results/deepwalk_ppi.npy           shape: (3852,  128)
    results/deepwalk_wikipedia.npy     shape: (4777,  128)

Runtime estimate:
    BlogCatalog  ~15-25 min
    PPI          ~5-10 min
    Wikipedia    ~8-15 min
"""

import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.deepwalk.deepwalk import DeepWalk


# ── Dataset config ─────────────────────────────────────────────────────────

DATASETS = {
    "blogcatalog": "data/processed/blogcatalog.gpickle",
    "ppi":         "data/processed/ppi.gpickle",
    "wikipedia":   "data/processed/wikipedia.gpickle",
}

# DeepWalk hyperparameters (from paper + team spec)
PARAMS = dict(
    embedding_dim = 128,
    walk_length   = 80,
    num_walks     = 10,
    window        = 10,
    workers       = 4,    # increase if your machine has more CPU cores
    epochs        = 1,
    seed          = 42,
)


# ── Runner ─────────────────────────────────────────────────────────────────

def run(dataset_name: str, graph_path: str):
    print(f"\n{'=' * 50}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 50}")

    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    model = DeepWalk(**PARAMS)
    model.fit(G)

    out_path = f"results/deepwalk_{dataset_name}.npy"
    model.save(out_path)
    return out_path


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    for name, path in DATASETS.items():
        run(name, path)

    print("\nAll embeddings saved. Hand .npy files to Aditya for classification.")
