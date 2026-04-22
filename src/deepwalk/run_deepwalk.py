"""
run_deepwalk.py
───────────────
Runs DeepWalk on all 3 datasets and saves embeddings to results/
Run from project root: python src/deepwalk/run_deepwalk.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from src.deepwalk.deepwalk import DeepWalk
from src.embeddings.graph_loader import load_graph

os.makedirs('results', exist_ok=True)

DATASETS = [
    { 'name': 'blogcatalog', 'graph': 'data/processed/blogcatalog.gpickle' },
    { 'name': 'ppi',         'graph': 'data/processed/ppi.gpickle' },
    { 'name': 'wikipedia',   'graph': 'data/processed/wikipedia.gpickle' },
]

for ds in DATASETS:
    print(f"\n{'='*55}")
    print(f"  Dataset: {ds['name'].upper()}")
    print(f"{'='*55}")

    G = load_graph(ds['graph'])

    model = DeepWalk(
        embedding_dim = 128,
        walk_length   = 80,
        num_walks     = 10,
        window        = 10,
        workers       = 4,
        seed          = 42,
    )

    model.fit(G)

    out_path = f"results/deepwalk_{ds['name']}.npy"
    model.save(out_path)

    sample_node = sorted(G.nodes())[0]
    emb = model.get_embedding(sample_node)
    print(f"[OK] node {sample_node} embedding shape: {emb.shape}")
    print(f"[OK] Full matrix: {model.embedding_shape()}")

print("\nAll DeepWalk embeddings saved to results/")
