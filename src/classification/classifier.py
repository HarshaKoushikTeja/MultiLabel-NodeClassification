"""
classifier.py  —  Aditya Khurana | feature/classification
Trains One-vs-Rest Logistic Regression classifiers on top of
DeepWalk and Node2Vec embeddings for multi-label node classification.

Supports: BlogCatalog, PPI, Wikipedia
Embedding models: DeepWalk, Node2Vec, Combined (DeepWalk + Node2Vec)

Usage:
    python classifier.py

Outputs (saved to results/):
    predictions_deepwalk_blogcatalog.npy
    predictions_deepwalk_ppi.npy
    predictions_deepwalk_wikipedia.npy
    predictions_node2vec_blogcatalog.npy
    predictions_node2vec_ppi.npy
    predictions_node2vec_wikipedia.npy
    predictions_combined_blogcatalog.npy
    predictions_combined_ppi.npy
    predictions_combined_wikipedia.npy

Also logs each run config to results/logs/ as a JSON file.
"""

import numpy as np
import pandas as pd
import pickle
import os
import json
import warnings
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")


# ── Config ────────────────────────────────────────────────────────────────────

DATASETS = {
    "blogcatalog": {
        "graph":       "data/processed/blogcatalog.gpickle",
        "labels":      "data/processed/labels_blogcatalog.csv",
        "train_idx":   "data/processed/train_idx.npy",
        "test_idx":    "data/processed/test_idx.npy",
        "idx_are_ids": True,    # BlogCatalog nodes are 1-indexed
    },
    "ppi": {
        "graph":       "data/processed/ppi.gpickle",
        "labels":      "data/processed/labels_ppi.csv",
        "train_idx":   "data/processed/train_idx_ppi.npy",
        "test_idx":    "data/processed/test_idx_ppi.npy",
        "idx_are_ids": False,
    },
    "wikipedia": {
        "graph":       "data/processed/wikipedia.gpickle",
        "labels":      "data/processed/labels_wikipedia.csv",
        "train_idx":   "data/processed/train_idx_wikipedia.npy",
        "test_idx":    "data/processed/test_idx_wikipedia.npy",
        "idx_are_ids": False,
    },
}

# Paths to embedding files produced by Shaman (DeepWalk) and Priyanshu (Node2Vec)
EMBEDDING_PATHS = {
    "deepwalk": {
        "blogcatalog": "results/deepwalk_blogcatalog.npy",
        "ppi":         "results/deepwalk_ppi.npy",
        "wikipedia":   "results/deepwalk_wikipedia.npy",
    },
    "node2vec": {
        "blogcatalog": "results/node2vec_blogcatalog.npy",
        "ppi":         "results/node2vec_ppi.npy",
        "wikipedia":   "results/node2vec_wikipedia.npy",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_graph(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_label_matrix(G, csv_path: str):
    """
    Build binary label matrix y aligned to sorted(G.nodes()).
    Returns y of shape [num_nodes x num_classes] and node_to_pos mapping.
    """
    nodes = sorted(G.nodes())
    node_to_pos = {n: i for i, n in enumerate(nodes)}

    df = pd.read_csv(csv_path).set_index("node_id")
    label_cols = df.columns.tolist()

    y = np.zeros((len(nodes), len(label_cols)), dtype=int)
    for node in nodes:
        if node in df.index:
            y[node_to_pos[node]] = df.loc[node, label_cols].values.astype(int)

    return y, node_to_pos


def resolve_indices(raw_idx: np.ndarray, node_to_pos: dict, idx_are_ids: bool) -> np.ndarray:
    """
    Convert raw index array to positional row indices.
    BlogCatalog uses node IDs (1-based); PPI and Wikipedia use positional indices.
    """
    if idx_are_ids:
        return np.array([node_to_pos[nid] for nid in raw_idx])
    return raw_idx


def load_embedding(path: str) -> np.ndarray:
    """Load embedding .npy file and cast to float64 for sklearn compatibility."""
    emb = np.load(path)
    return emb.astype(np.float64)


def train_and_predict(X_train, y_train, X_test):
    """Train OvR Logistic Regression and return predictions."""
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, solver="lbfgs"),
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def save_log(log: dict, dataset: str, emb_name: str):
    """Save run config and metrics to results/logs/ as JSON."""
    os.makedirs("results/logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"results/logs/{emb_name}_{dataset}_{timestamp}.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  Log saved  -> {log_path}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_classifier(emb_name: str, dataset: str, X: np.ndarray, cfg: dict):
    """
    Train classifier on embedding X for one dataset and save predictions.

    Parameters
    ----------
    emb_name : str
        One of 'deepwalk', 'node2vec', 'combined'
    dataset  : str
        One of 'blogcatalog', 'ppi', 'wikipedia'
    X        : np.ndarray
        Full embedding matrix, shape [num_nodes x embedding_dim]
    cfg      : dict
        Dataset config entry from DATASETS
    """
    print(f"\n  Embedding : {emb_name}  |  Dataset: {dataset}")
    print(f"  X shape   : {X.shape}")

    G = load_graph(cfg["graph"])
    y, node_to_pos = get_label_matrix(G, cfg["labels"])

    train_pos = resolve_indices(
        np.load(cfg["train_idx"], allow_pickle=True), node_to_pos, cfg["idx_are_ids"]
    )
    test_pos = resolve_indices(
        np.load(cfg["test_idx"], allow_pickle=True), node_to_pos, cfg["idx_are_ids"]
    )

    X_train, X_test = X[train_pos], X[test_pos]
    y_train, y_test = y[train_pos], y[test_pos]

    y_pred = train_and_predict(X_train, y_train, X_test)

    # Quick metrics for console feedback
    micro = f1_score(y_test, y_pred, average="micro", zero_division=0)
    macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"  Micro-F1  : {micro:.4f}  |  Macro-F1: {macro:.4f}")

    # Save predictions
    os.makedirs("results", exist_ok=True)
    out_path = f"results/predictions_{emb_name}_{dataset}.npy"
    np.save(out_path, y_pred)
    print(f"  Saved      -> {out_path}  shape: {y_pred.shape}")

    # Save log
    save_log({
        "embedding":   emb_name,
        "dataset":     dataset,
        "x_shape":     list(X.shape),
        "train_size":  int(len(train_pos)),
        "test_size":   int(len(test_pos)),
        "num_classes": int(y.shape[1]),
        "micro_f1":    round(micro, 4),
        "macro_f1":    round(macro, 4),
        "model":       "OneVsRestClassifier(LogisticRegression)",
        "solver":      "lbfgs",
        "max_iter":    1000,
    }, dataset, emb_name)

    return y_pred


def main():
    print("=" * 55)
    print("Embedding Classifier — Aditya Khurana")
    print("=" * 55)

    for dataset, cfg in DATASETS.items():
        print(f"\n{'='*55}\nDataset: {dataset.upper()}\n{'='*55}")

        # Load both embedding matrices
        dw_path = EMBEDDING_PATHS["deepwalk"][dataset]
        n2v_path = EMBEDDING_PATHS["node2vec"][dataset]

        X_dw  = load_embedding(dw_path)
        X_n2v = load_embedding(n2v_path)

        # DeepWalk classifier
        run_classifier("deepwalk", dataset, X_dw, cfg)

        # Node2Vec classifier
        run_classifier("node2vec", dataset, X_n2v, cfg)

        # Combined classifier — concatenate both embeddings [num_nodes x 256]
        X_combined = np.hstack([X_dw, X_n2v])
        run_classifier("combined", dataset, X_combined, cfg)

    print(f"\n{'='*55}")
    print("All predictions saved to results/")
    print("Pass .npy files to Sagar's metrics.py for full evaluation.")


if __name__ == "__main__":
    main()
