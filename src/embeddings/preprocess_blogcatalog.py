"""
preprocess_blogcatalog.py
─────────────────────────
This should Convert the raw 'BlogCatalog CSVs' → processed files for graph_loader.py

Outputs should be this:
  data/processed/blogcatalog.gpickle
  data/processed/labels_blogcatalog.csv
  data/processed/train_idx.npy
  data/processed/test_idx.npy
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.embeddings.graph_loader import save_graph, load_graph, graph_stats

# ── Paths ──────────────────────────────────────────────────────────────
RAW_DIR       = 'data/processed/BlogCatalog-dataset/data'
PROCESSED_DIR = 'data/processed'
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Step 1: Load raw files (no headers) ───────────────────────────────
edges_df       = pd.read_csv(os.path.join(RAW_DIR, 'edges.csv'),
                              header=None, names=['node1', 'node2'])
group_edges_df = pd.read_csv(os.path.join(RAW_DIR, 'group-edges.csv'),
                              header=None, names=['node_id', 'group_id'])
nodes_df       = pd.read_csv(os.path.join(RAW_DIR, 'nodes.csv'),
                              header=None, names=['node_id'])
groups_df      = pd.read_csv(os.path.join(RAW_DIR, 'groups.csv'),
                              header=None, names=['group_id'])

print(f"Edges     : {len(edges_df)}")
print(f"Nodes     : {len(nodes_df)}")
print(f"Groups    : {len(groups_df)}")
print(f"Node-Group: {len(group_edges_df)}")

# ── Step 2: Build NetworkX graph ───────────────────────────────────────
G = nx.from_pandas_edgelist(
    edges_df,
    source='node1',
    target='node2',
    create_using=nx.Graph()
)

# Add any isolated nodes present in nodes.csv but not in edges
for node in nodes_df['node_id']:
    G.add_node(int(node))

graph_stats(G)

# ── Step 3: Save graph as .gpickle ────────────────────────────────────
save_graph(G, os.path.join(PROCESSED_DIR, 'blogcatalog.gpickle'))

# ── Step 4: Build binary label matrix ─────────────────────────────────
node_list   = sorted(G.nodes())                      # fixed ordering
num_nodes   = len(node_list)
num_classes = groups_df['group_id'].max()            # 39 for BlogCatalog
node_index  = {n: i for i, n in enumerate(node_list)}

label_matrix = np.zeros((num_nodes, num_classes), dtype=int)

for _, row in group_edges_df.iterrows():
    node  = int(row['node_id'])
    group = int(row['group_id']) - 1               # 0-indexed
    if node in node_index:
        label_matrix[node_index[node]][group] = 1

print(f"\nLabel matrix shape : {label_matrix.shape}")
print(f"Avg labels per node: {label_matrix.sum(axis=1).mean():.2f}")
print(f"Nodes with 0 labels: {(label_matrix.sum(axis=1) == 0).sum()}")

# ── Step 5: Save label CSV (format Harsha's load_labels() expects) ────
label_cols = [f'label_{i}' for i in range(num_classes)]
label_df   = pd.DataFrame(label_matrix, columns=label_cols)
label_df.insert(0, 'node_id', node_list)
label_df.to_csv(os.path.join(PROCESSED_DIR, 'labels_blogcatalog.csv'), index=False)
print(f"Labels saved → data/processed/labels_blogcatalog.csv")

# ── Step 6: Train/test split (80/20) ──────────────────────────────────
node_array = np.array(node_list)
np.random.seed(42)
shuffled   = node_array.copy()
np.random.shuffle(shuffled)

split      = int(0.8 * len(shuffled))
train_idx  = shuffled[:split]
test_idx   = shuffled[split:]

np.save(os.path.join(PROCESSED_DIR, 'train_idx.npy'), train_idx)
np.save(os.path.join(PROCESSED_DIR, 'test_idx.npy'),  test_idx)
print(f"Split saved → train: {len(train_idx)}, test: {len(test_idx)}")

# ── Step 7: Verify everything loads correctly ──────────────────────────
print("\n── Verification ──────────────────────────────────────")
G_check      = load_graph(os.path.join(PROCESSED_DIR, 'blogcatalog.gpickle'))
train_check  = np.load(os.path.join(PROCESSED_DIR, 'train_idx.npy'))
test_check   = np.load(os.path.join(PROCESSED_DIR, 'test_idx.npy'))
labels_check = pd.read_csv(os.path.join(PROCESSED_DIR, 'labels_blogcatalog.csv'))

print(f"✅ Graph     : {G_check.number_of_nodes()} nodes, {G_check.number_of_edges()} edges")
print(f"✅ Labels    : {labels_check.shape}")
print(f"✅ Train idx : {len(train_check)}")
print(f"✅ Test idx  : {len(test_check)}")
print("\n🎉 All files ready. Teammates can now call GraphLoader.load_processed()")