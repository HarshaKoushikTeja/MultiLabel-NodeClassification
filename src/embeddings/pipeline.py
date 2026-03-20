"""
pipeline.py
───────────
Integration pipeline — Harsha's core deliverable.

This script ties together:
  1. Graph loading       (graph_loader.py)
  2. DeepWalk embeddings (src/deepwalk/)
  3. Node2Vec embeddings (src/node2vec/)
  4. Output matrices     → ready for Aditya's classifier

Run this AFTER Shaman and Priyanshu have saved their .npy files.

Usage
-----
    python src/embeddings/pipeline.py \
        --dataset blogcatalog \
        --dw_path results/deepwalk_blogcatalog.npy \
        --n2v_path results/node2vec_blogcatalog.npy \
        --graph_path data/processed/blogcatalog.gpickle \
        --labels_path data/processed/labels_blogcatalog.csv

Output
------
    results/pipeline_blogcatalog_deepwalk.npy   ← embedding matrix (DeepWalk)
    results/pipeline_blogcatalog_node2vec.npy   ← embedding matrix (Node2Vec)
    results/pipeline_blogcatalog_combined.npy   ← concatenated DW + N2V
    results/pipeline_blogcatalog_labels.npy     ← aligned label matrix
"""

import os
import argparse
import numpy as np
from graph_loader import load_graph, load_labels, graph_stats


# ──────────────────────────────────────────────────────────────────────────────
# Core pipeline function
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    graph_path:  str,
    labels_path: str,
    dw_path:     str,
    n2v_path:    str,
    dataset:     str,
    output_dir:  str = "results",
) -> dict:
    """
    Load embeddings from both models and align them with node labels.

    Parameters
    ----------
    graph_path  : str  — path to the .gpickle graph file
    labels_path : str  — path to the labels CSV
    dw_path     : str  — path to DeepWalk .npy embedding file
    n2v_path    : str  — path to Node2Vec .npy embedding file
    dataset     : str  — dataset name (e.g. 'blogcatalog', 'ppi')
    output_dir  : str  — directory to save output files

    Returns
    -------
    dict with keys:
        'deepwalk'  : np.ndarray [num_nodes, dw_dim]
        'node2vec'  : np.ndarray [num_nodes, n2v_dim]
        'combined'  : np.ndarray [num_nodes, dw_dim + n2v_dim]
        'labels'    : np.ndarray [num_nodes, num_classes]
        'node_order': list of node IDs (row order for all matrices above)
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load graph
    print(f"\n{'='*50}")
    print(f"Pipeline: dataset = {dataset}")
    print(f"{'='*50}")
    graph = load_graph(graph_path)
    graph_stats(graph)

    # 2. Load labels
    labels_dict = load_labels(labels_path)

    # 3. Determine node order (sorted for reproducibility)
    node_order = sorted(graph.nodes())
    num_nodes = len(node_order)
    print(f"Node order established: {num_nodes} nodes")

    # 4. Load DeepWalk embeddings
    print(f"\nLoading DeepWalk embeddings from: {dw_path}")
    dw_embeddings = np.load(dw_path)
    print(f"  Shape: {dw_embeddings.shape}")

    # 5. Load Node2Vec embeddings
    print(f"Loading Node2Vec embeddings from: {n2v_path}")
    n2v_embeddings = np.load(n2v_path)
    print(f"  Shape: {n2v_embeddings.shape}")

    # 6. Validate shapes
    assert dw_embeddings.shape[0] == num_nodes, (
        f"DeepWalk shape mismatch: {dw_embeddings.shape[0]} rows "
        f"but graph has {num_nodes} nodes"
    )
    assert n2v_embeddings.shape[0] == num_nodes, (
        f"Node2Vec shape mismatch: {n2v_embeddings.shape[0]} rows "
        f"but graph has {num_nodes} nodes"
    )

    # 7. Build aligned label matrix
    num_classes = len(next(iter(labels_dict.values())))
    label_matrix = np.zeros((num_nodes, num_classes), dtype=int)
    for i, node_id in enumerate(node_order):
        if node_id in labels_dict:
            label_matrix[i] = labels_dict[node_id]
    print(f"\nLabel matrix shape: {label_matrix.shape}")

    # 8. Concatenate embeddings (for combined experiment)
    combined = np.hstack([dw_embeddings, n2v_embeddings])
    print(f"Combined embedding shape: {combined.shape}")

    # 9. Save all outputs
    dw_out  = os.path.join(output_dir, f"pipeline_{dataset}_deepwalk.npy")
    n2v_out = os.path.join(output_dir, f"pipeline_{dataset}_node2vec.npy")
    comb_out= os.path.join(output_dir, f"pipeline_{dataset}_combined.npy")
    lbl_out = os.path.join(output_dir, f"pipeline_{dataset}_labels.npy")

    np.save(dw_out,   dw_embeddings)
    np.save(n2v_out,  n2v_embeddings)
    np.save(comb_out, combined)
    np.save(lbl_out,  label_matrix)

    print(f"\n✓ Saved DeepWalk output  → {dw_out}")
    print(f"✓ Saved Node2Vec output  → {n2v_out}")
    print(f"✓ Saved combined output  → {comb_out}")
    print(f"✓ Saved labels           → {lbl_out}")
    print(f"\nPipeline complete for dataset: {dataset}")

    return {
        "deepwalk":   dw_embeddings,
        "node2vec":   n2v_embeddings,
        "combined":   combined,
        "labels":     label_matrix,
        "node_order": node_order,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Integration pipeline for P15 graph embedding project."
    )
    parser.add_argument("--dataset",      required=True, help="Dataset name: blogcatalog or ppi")
    parser.add_argument("--graph_path",   required=True, help="Path to .gpickle graph file")
    parser.add_argument("--labels_path",  required=True, help="Path to labels CSV file")
    parser.add_argument("--dw_path",      required=True, help="Path to DeepWalk .npy file")
    parser.add_argument("--n2v_path",     required=True, help="Path to Node2Vec .npy file")
    parser.add_argument("--output_dir",   default="results", help="Output directory (default: results/)")
    args = parser.parse_args()

    run_pipeline(
        graph_path  = args.graph_path,
        labels_path = args.labels_path,
        dw_path     = args.dw_path,
        n2v_path    = args.n2v_path,
        dataset     = args.dataset,
        output_dir  = args.output_dir,
    )
