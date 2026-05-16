"""
data_loader.py — GNN Extension
Converts existing .gpickle graphs + label CSVs into PyTorch Geometric Data objects.
Uses the SAME graphs, labels, and train/test splits as DeepWalk/Node2Vec
so GNN results are directly comparable.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


DATASETS = {
    "blogcatalog": {
        "graph":     "data/processed/blogcatalog.gpickle",
        "labels":    "data/processed/labels_blogcatalog.csv",
        "train_idx": "data/processed/train_idx.npy",
        "test_idx":  "data/processed/test_idx.npy",
        "idx_are_ids": True,   # BlogCatalog node IDs are 1-indexed
    },
    "ppi": {
        "graph":     "data/processed/ppi.gpickle",
        "labels":    "data/processed/labels_ppi.csv",
        "train_idx": "data/processed/train_idx_ppi.npy",
        "test_idx":  "data/processed/test_idx_ppi.npy",
        "idx_are_ids": False,
    },
    "wikipedia": {
        "graph":     "data/processed/wikipedia.gpickle",
        "labels":    "data/processed/labels_wikipedia.csv",
        "train_idx": "data/processed/train_idx_wikipedia.npy",
        "test_idx":  "data/processed/test_idx_wikipedia.npy",
        "idx_are_ids": False,
    },
}


def load_graph(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_indices(raw_idx, node_to_pos, idx_are_ids):
    """Convert raw index array to positional row indices."""
    if idx_are_ids:
        return np.array([node_to_pos[nid] for nid in raw_idx])
    return raw_idx


def build_pyg_data(dataset_name):
    """
    Build a PyTorch Geometric Data object for the given dataset.

    Returns
    -------
    data : torch_geometric.data.Data
        Contains x (node features), edge_index, y (multi-label targets),
        train_mask, test_mask.
    """
    cfg = DATASETS[dataset_name]

    G = load_graph(cfg["graph"])
    nodes = sorted(G.nodes())
    node_to_pos = {n: i for i, n in enumerate(nodes)}
    num_nodes = len(nodes)

    # ── Edge index: [2, num_edges*2] (undirected → both directions) ──
    edges = []
    for u, v in G.edges():
        edges.append([node_to_pos[u], node_to_pos[v]])
        edges.append([node_to_pos[v], node_to_pos[u]])  # reverse
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # ── Labels: binary multi-label matrix [num_nodes, num_classes] ──
    df = pd.read_csv(cfg["labels"]).set_index("node_id")
    label_cols = df.columns.tolist()
    y = np.zeros((num_nodes, len(label_cols)), dtype=np.float32)
    for node in nodes:
        if node in df.index:
            y[node_to_pos[node]] = df.loc[node, label_cols].values.astype(np.float32)
    y = torch.tensor(y, dtype=torch.float)

    # ── Node features ──
    # GNNs need input features. Since these datasets have no natural node
    # features, we use an identity-like structural feature: node degree
    # plus a learnable embedding handled in the model. Here we provide
    # degree as a simple 1-D feature, expanded to a small feature vector.
    deg = np.array([G.degree(n) for n in nodes], dtype=np.float32).reshape(-1, 1)
    # Normalize degree (log scale, then standardize)
    deg = np.log1p(deg)
    deg = (deg - deg.mean()) / (deg.std() + 1e-8)
    x = torch.tensor(deg, dtype=torch.float)

    # ── Train / test masks ──
    train_pos = resolve_indices(
        np.load(cfg["train_idx"], allow_pickle=True), node_to_pos, cfg["idx_are_ids"]
    )
    test_pos = resolve_indices(
        np.load(cfg["test_idx"], allow_pickle=True), node_to_pos, cfg["idx_are_ids"]
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_pos] = True
    test_mask[test_pos] = True

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.test_mask = test_mask
    data.num_classes = len(label_cols)

    return data


if __name__ == "__main__":
    for name in DATASETS:
        d = build_pyg_data(name)
        print(f"\n{name.upper()}")
        print(f"  Nodes: {d.num_nodes}  |  Edges: {d.edge_index.shape[1]}")
        print(f"  Features: {d.x.shape}  |  Classes: {d.num_classes}")
        print(f"  Train: {d.train_mask.sum().item()}  |  Test: {d.test_mask.sum().item()}")