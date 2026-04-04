"""
run_node2vec.py
───────────────
Runs Node2Vec on all three datasets and saves embeddings to results/.

Run from project root:
    python src/node2vec/run_node2vec.py
"""

import os
import sys
sys.path.append(os.path.abspath('.'))

from src.node2vec.node2vec import Node2Vec
from src.embeddings.graph_loader import load_graph

os.makedirs('results', exist_ok=True)

DATASETS = [
    {
        'name'  : 'blogcatalog',
        'graph' : 'data/processed/blogcatalog.gpickle',
        'p'     : 1, 'q': 1,    # start with unbiased (same as DeepWalk)
    },
    {
        'name'  : 'ppi',
        'graph' : 'data/processed/ppi.gpickle',
        'p'     : 1, 'q': 0.5,  # slightly inward bias for biological network
    },
    {
        'name'  : 'wikipedia',
        'graph' : 'data/processed/wikipedia.gpickle',
        'p'     : 1, 'q': 2,    # outward bias for text network
    },
]

for ds in DATASETS:
    print(f"\n{'='*55}")
    print(f"  Dataset: {ds['name'].upper()}")
    print(f"{'='*55}")

    G     = load_graph(ds['graph'])
    model = Node2Vec(
        embedding_dim = 128,
        walk_length   = 80,
        num_walks     = 10,
        p             = ds['p'],
        q             = ds['q'],
        workers       = 4,
        seed          = 42,
    )

    model.fit(G)

    out_path = f"results/node2vec_{ds['name']}.npy"
    model.save(out_path)

    # Quick sanity check
    sample_node = sorted(G.nodes())[0]
    emb         = model.get_embedding(sample_node)
    print(f"[✅] Sample embedding — node {sample_node}: shape {emb.shape}, "
          f"mean {emb.mean():.4f}")
    print(f"[✅] Full matrix: {model.embedding_shape()}")

print("\n🎉 All Node2Vec embeddings saved to results/")