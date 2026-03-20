"""
graph_loader.py
───────────────
Unified graph loader for all datasets in the P15 project.

This module is the single entry point for loading any dataset into a
consistent NetworkX graph object. All downstream code (DeepWalk,
Node2Vec, baseline classifiers) should use this loader — never load
data directly in the model files.

Supported datasets
------------------
- BlogCatalog  (social network, multi-label)
- PPI          (protein-protein interaction, multi-label)

Usage
-----
    from src.embeddings.graph_loader import load_graph, load_labels

    graph  = load_graph('data/processed/blogcatalog.gpickle')
    labels = load_labels('data/processed/labels_blogcatalog.csv')
"""

import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from typing import Tuple, Dict, List


# ──────────────────────────────────────────────────────────────────────────────
# Primary loader — call this from all model files
# ──────────────────────────────────────────────────────────────────────────────

def load_graph(filepath: str) -> nx.Graph:
    """
    Load a graph from a .gpickle or edge-list file into a NetworkX Graph.

    Parameters
    ----------
    filepath : str
        Path to the graph file. Supported formats:
        - .gpickle   : saved NetworkX graph (fastest)
        - .edgelist  : plain text edge list (space separated)
        - .txt       : same as edgelist

    Returns
    -------
    nx.Graph
        Undirected graph with integer node IDs.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file format is not supported.

    Example
    -------
    graph = load_graph('data/processed/blogcatalog.gpickle')
    print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Graph file not found: {filepath}")

    ext = os.path.splitext(filepath)[-1].lower()

    if ext == ".gpickle":
        with open(filepath, "rb") as f:
            graph = pickle.load(f)

    elif ext in (".edgelist", ".txt"):
        graph = nx.read_edgelist(filepath, nodetype=int)

    else:
        raise ValueError(
            f"Unsupported file format: '{ext}'. "
            "Use .gpickle or .edgelist"
        )

    # Ensure the graph is undirected
    if isinstance(graph, nx.DiGraph):
        graph = graph.to_undirected()

    print(
        f"Loaded graph: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges  [{filepath}]"
    )
    return graph


def save_graph(graph: nx.Graph, filepath: str) -> None:
    """
    Save a NetworkX graph to a .gpickle file.

    Prashant should use this after loading raw edge lists.

    Parameters
    ----------
    graph    : nx.Graph  — the graph to save
    filepath : str       — destination path (e.g. 'data/processed/ppi.gpickle')
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(graph, f)
    print(f"Graph saved to: {filepath}")


# ──────────────────────────────────────────────────────────────────────────────
# Label loader
# ──────────────────────────────────────────────────────────────────────────────

def load_labels(filepath: str) -> Dict[int, List[int]]:
    """
    Load multi-label node labels from a CSV file.

    Expected CSV format (no header row needed, or with 'node_id' header):
        node_id, label1, label2, ...
        0, 1, 0, 1, 0, ...
        1, 0, 0, 0, 1, ...

    Parameters
    ----------
    filepath : str
        Path to the labels CSV file.

    Returns
    -------
    dict
        {node_id (int): label_vector (list of int 0/1)}

    Example
    -------
    labels = load_labels('data/processed/labels_blogcatalog.csv')
    print(labels[0])   # e.g. [1, 0, 1, 0, ...]
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Labels file not found: {filepath}")

    df = pd.read_csv(filepath)
    node_col = df.columns[0]
    label_cols = df.columns[1:]

    labels = {}
    for _, row in df.iterrows():
        node_id = int(row[node_col])
        labels[node_id] = [int(row[c]) for c in label_cols]

    print(
        f"Loaded labels: {len(labels)} nodes, "
        f"{len(label_cols)} classes  [{filepath}]"
    )
    return labels


# ──────────────────────────────────────────────────────────────────────────────
# Train/test split loader
# ──────────────────────────────────────────────────────────────────────────────

def load_splits(train_path: str, test_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load pre-computed train and test node index arrays.

    Prashant saves these as .npy files. This function loads them.

    Parameters
    ----------
    train_path : str   — e.g. 'data/processed/train_idx.npy'
    test_path  : str   — e.g. 'data/processed/test_idx.npy'

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (train_indices, test_indices)
    """
    train_idx = np.load(train_path)
    test_idx  = np.load(test_path)
    print(f"Train nodes: {len(train_idx)}, Test nodes: {len(test_idx)}")
    return train_idx, test_idx


# ──────────────────────────────────────────────────────────────────────────────
# Graph statistics helper (for EDA — Sagar/Prashant)
# ──────────────────────────────────────────────────────────────────────────────

def graph_stats(graph: nx.Graph) -> dict:
    """
    Print and return basic statistics about a graph.

    Parameters
    ----------
    graph : nx.Graph

    Returns
    -------
    dict
        Dictionary of graph statistics.
    """
    degrees = [d for _, d in graph.degree()]
    stats = {
        "num_nodes":   graph.number_of_nodes(),
        "num_edges":   graph.number_of_edges(),
        "avg_degree":  round(sum(degrees) / len(degrees), 2),
        "max_degree":  max(degrees),
        "min_degree":  min(degrees),
        "is_connected": nx.is_connected(graph),
        "density":     round(nx.density(graph), 6),
    }
    print("\n── Graph Statistics ──────────────────────")
    for k, v in stats.items():
        print(f"  {k:<18}: {v}")
    print("──────────────────────────────────────────\n")
    return stats
