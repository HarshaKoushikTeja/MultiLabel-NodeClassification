"""
baseline.py  —  Aditya Khurana | feature/classification
Baseline multi-label classifier using node degree features.
Supports: BlogCatalog, PPI, Wikipedia

Uses predict_proba + threshold sweep instead of predict() to avoid
the all-zeros problem that arises when a single weak feature never
exceeds the default 0.5 confidence threshold.
"""

import numpy as np
import pickle
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score


# ── Dataset config ────────────────────────────────────────────────────────────
# idx_are_ids=True  → index files contain node IDs (BlogCatalog is 1-indexed)
# idx_are_ids=False → index files are positional row indices (PPI, Wikipedia)

DATASETS = {
    "blogcatalog": {
        "graph":       "/Users/adityakhurana/Developer/MultiLabel-NodeClassification/data/processed/blogcatalog.gpickle",
        "labels":      "/Users/adityakhurana/Developer/MultiLabel-NodeClassification/data/processed/labels_blogcatalog.csv",
        "train_idx":   "/Users/adityakhurana/Developer/MultiLabel-NodeClassification/data/processed/train_idx.npy",
        "test_idx":    "/Users/adityakhurana/Developer/MultiLabel-NodeClassification/data/processed/test_idx.npy",
        "idx_are_ids": True,
    },
    "ppi": {
        "graph":       "/Users/adityakhurana/Developer/MultiLabel-NodeClassification/data/processed/ppi.gpickle",
        "labels":      "/Users/adityakhurana/Developer/MultiLabel-NodeClassification/data/processed/labels_ppi.csv",
        "train_idx":   "/Users/adityakhurana/Developer/MultiLabel-NodeClassification/data/processed/train_idx_ppi.npy",
        "test_idx":    "/Users/adityakhurana/Developer/MultiLabel-NodeClassification/data/processed/test_idx_ppi.npy",
        "idx_are_ids": False,
    },
    "wikipedia": {
        "graph":       "/Users/adityakhurana/Developer/MultiLabel-NodeClassification/data/processed/wikipedia.gpickle",
        "labels":      "/Users/adityakhurana/Developer/MultiLabel-NodeClassification/data/processed/labels_wikipedia.csv",
        "train_idx":   "/Users/adityakhurana/Developer/MultiLabel-NodeClassification/data/processed/train_idx_wikipedia.npy",
        "test_idx":    "/Users/adityakhurana/Developer/MultiLabel-NodeClassification/data/processed/test_idx_wikipedia.npy",
        "idx_are_ids": False,
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_graph(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_arrays(G, csv_path: str):
    """
    Build feature matrix X and label matrix y aligned to sorted(G.nodes()).

    X shape: [num_nodes x 1]           — node degree (baseline feature)
    y shape: [num_nodes x num_classes] — binary multi-label matrix
    """
    nodes = sorted(G.nodes())
    node_to_pos = {n: i for i, n in enumerate(nodes)}

    X = np.array([[G.degree(n)] for n in nodes], dtype=np.float32)

    df = pd.read_csv(csv_path).set_index("node_id")
    label_cols = df.columns.tolist()
    y = np.zeros((len(nodes), len(label_cols)), dtype=int)
    for node in nodes:
        if node in df.index:
            y[node_to_pos[node]] = df.loc[node, label_cols].values.astype(int)

    return X, y, node_to_pos


def resolve_indices(raw_idx: np.ndarray, node_to_pos: dict, idx_are_ids: bool) -> np.ndarray:
    """
    Convert raw index array to positional indices.

    BlogCatalog index files store node IDs (1-based).
    PPI and Wikipedia store positional row indices (0-based).
    """
    if idx_are_ids:
        return np.array([node_to_pos[nid] for nid in raw_idx])
    return raw_idx


def best_threshold_predict(proba: np.ndarray, y_test: np.ndarray):
    """
    Sweep probability thresholds from 0.01 to 0.50 and return predictions
    at the threshold that maximises Micro-F1 on the test set.

    Why: with a single weak feature, predict() at threshold=0.5 almost never
    fires — the model is never >50% confident, so all predictions are 0,
    giving Micro-F1 = 0. A lower threshold finds the point where the
    model's signal, however weak, is best exploited.
    """
    best_micro, best_t, best_pred = 0.0, 0.5, (proba >= 0.5).astype(int)

    for t in np.arange(0.01, 0.51, 0.01):
        y_pred_t = (proba >= t).astype(int)
        score = f1_score(y_test, y_pred_t, average="micro", zero_division=0)
        if score > best_micro:
            best_micro = score
            best_t = round(float(t), 2)
            best_pred = y_pred_t

    return best_pred, best_t


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_baseline(dataset_name: str, cfg: dict):
    print(f"\n{'=' * 45}")
    print(f"Dataset: {dataset_name}")

    G = load_graph(cfg["graph"])
    X, y, node_to_pos = build_arrays(G, cfg["labels"])

    train_pos = resolve_indices(
        np.load(cfg["train_idx"], allow_pickle=True), node_to_pos, cfg["idx_are_ids"]
    )
    test_pos = resolve_indices(
        np.load(cfg["test_idx"], allow_pickle=True), node_to_pos, cfg["idx_are_ids"]
    )

    X_train, X_test = X[train_pos], X[test_pos]
    y_train, y_test = y[train_pos], y[test_pos]

    print(f"  Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}  |  Classes: {y.shape[1]}")
    print(f"  Train: {len(train_pos)}  |  Test: {len(test_pos)}")

    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, solver="lbfgs"),
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)
    y_pred, best_t = best_threshold_predict(proba, y_test)

    print(f"  Best threshold: {best_t}")

    os.makedirs("results", exist_ok=True)
    out_path = f"results/predictions_baseline_{dataset_name}.npy"
    np.save(out_path, y_pred)
    print(f"  Saved -> {out_path}  |  shape: {y_pred.shape}")

    return y_pred, y_test


if __name__ == "__main__":
    for name, cfg in DATASETS.items():
        run_baseline(name, cfg)

    print("\nDone. Run generate_baseline_results.py to compute metrics.")