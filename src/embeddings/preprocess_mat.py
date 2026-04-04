"""
preprocess_mat.py
─────────────────
Converts PPI and Wikipedia .mat files into processed files
matching the same format as BlogCatalog outputs.

Outputs for PPI:
  data/processed/ppi.gpickle
  data/processed/labels_ppi.csv
  data/processed/train_idx_ppi.npy
  data/processed/test_idx_ppi.npy

Outputs for Wikipedia:
  data/processed/wikipedia.gpickle
  data/processed/labels_wikipedia.csv
  data/processed/train_idx_wikipedia.npy
  data/processed/test_idx_wikipedia.npy
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import scipy.io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.embeddings.graph_loader import save_graph, load_graph, graph_stats

PROCESSED_DIR = 'data/processed'
os.makedirs(PROCESSED_DIR, exist_ok=True)


def preprocess_mat(mat_path, dataset_name):
    """
    Generic processor for .mat datasets with 'network' and 'group' keys.
    Automatically extracts Largest Connected Component if graph is disconnected.

    Parameters
    ----------
    mat_path     : str  — path to .mat file
    dataset_name : str  — short name e.g. 'ppi', 'wikipedia'
    """
    print(f"\n{'='*50}")
    print(f"  Processing: {dataset_name.upper()}")
    print(f"{'='*50}")

    # ── Step 1: Load .mat file ─────────────────────────────────────────
    mat     = scipy.io.loadmat(mat_path)
    network = mat['network']   # sparse adjacency matrix (N x N)
    group   = mat['group']     # label matrix (N x C)

    # ── Step 2: Build full NetworkX graph ──────────────────────────────
    cx     = network.tocoo()
    G_full = nx.Graph()
    G_full.add_nodes_from(range(network.shape[0]))   # nodes 0 to N-1
    edges  = [(int(i), int(j)) for i, j in zip(cx.row, cx.col) if i < j]
    G_full.add_edges_from(edges)

    print(f"\nFull graph stats:")
    graph_stats(G_full)

    # ── Step 3: Extract Largest Connected Component if disconnected ────
    if not nx.is_connected(G_full):
        lcc_nodes = max(nx.connected_components(G_full), key=len)
        G_lcc     = G_full.subgraph(lcc_nodes).copy()

        # Keep original node IDs before remapping (needed for label alignment)
        original_nodes_kept = sorted(G_lcc.nodes())

        # Remap node IDs to clean 0-indexed integers
        mapping = {old: new for new, old in enumerate(original_nodes_kept)}
        G       = nx.relabel_nodes(G_lcc, mapping)

        print(f"\n⚠️  Graph not connected!")
        print(f"   Original : {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges")
        print(f"   LCC kept : {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"   Dropped  : {G_full.number_of_nodes() - G.number_of_nodes()} isolated nodes\n")
    else:
        G                   = G_full
        original_nodes_kept = sorted(G.nodes())   # identity — no change
        print(f"\n✅ Graph is connected — no LCC extraction needed\n")

    print(f"Final graph stats:")
    graph_stats(G)

    # ── Step 4: Save graph ─────────────────────────────────────────────
    graph_path = os.path.join(PROCESSED_DIR, f'{dataset_name}.gpickle')
    save_graph(G, graph_path)

    # ── Step 5: Build label matrix aligned to kept nodes ──────────────
    if hasattr(group, 'toarray'):
        label_matrix_full = group.toarray().astype(int)
    else:
        label_matrix_full = np.array(group).astype(int)

    # Filter rows to only nodes in LCC, in correct order
    label_matrix = label_matrix_full[original_nodes_kept]
    node_list    = sorted(G.nodes())   # clean 0-indexed
    num_classes  = label_matrix.shape[1]

    print(f"Label matrix shape : {label_matrix.shape}")
    print(f"Avg labels per node: {label_matrix.sum(axis=1).mean():.2f}")
    print(f"Nodes with 0 labels: {(label_matrix.sum(axis=1) == 0).sum()}")

    # ── Step 6: Save label CSV ─────────────────────────────────────────
    label_cols = [f'label_{i}' for i in range(num_classes)]
    label_df   = pd.DataFrame(label_matrix, columns=label_cols)
    label_df.insert(0, 'node_id', node_list)

    label_path = os.path.join(PROCESSED_DIR, f'labels_{dataset_name}.csv')
    label_df.to_csv(label_path, index=False)
    print(f"Labels saved → {label_path}")

    # ── Step 7: Train/test split (80/20) ──────────────────────────────
    node_array = np.array(node_list)
    np.random.seed(42)
    shuffled   = node_array.copy()
    np.random.shuffle(shuffled)

    split     = int(0.8 * len(shuffled))
    train_idx = shuffled[:split]
    test_idx  = shuffled[split:]

    np.save(os.path.join(PROCESSED_DIR, f'train_idx_{dataset_name}.npy'), train_idx)
    np.save(os.path.join(PROCESSED_DIR, f'test_idx_{dataset_name}.npy'),  test_idx)
    print(f"Split saved → train: {len(train_idx)}, test: {len(test_idx)}")

    # ── Step 8: Verify everything ──────────────────────────────────────
    print(f"\n── Verification ───────────────────────────────────────")
    G_check     = load_graph(graph_path)
    train_check = np.load(os.path.join(PROCESSED_DIR, f'train_idx_{dataset_name}.npy'))
    test_check  = np.load(os.path.join(PROCESSED_DIR, f'test_idx_{dataset_name}.npy'))

    print(f"✅ Graph  : {G_check.number_of_nodes()} nodes, {G_check.number_of_edges()} edges")
    print(f"✅ Labels : {label_df.shape}  (includes node_id column)")
    print(f"✅ Train  : {len(train_check)}  |  Test: {len(test_check)}")
    assert G_check.number_of_nodes() == label_matrix.shape[0], \
        "❌ Mismatch: graph nodes != label rows!"
    assert len(train_check) + len(test_check) == G_check.number_of_nodes(), \
        "❌ Mismatch: train+test != total nodes!"
    print(f"🎉 {dataset_name.upper()} ready!\n")

    return G, label_matrix, node_list


# ── Run both datasets ──────────────────────────────────────────────────
if __name__ == '__main__':
    preprocess_mat(
        mat_path     = 'data/processed/Protein_Protein_Interaction.mat',
        dataset_name = 'ppi'
    )
    preprocess_mat(
        mat_path     = 'data/processed/Wikipedia.mat',
        dataset_name = 'wikipedia'
    )