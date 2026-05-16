"""
models.py — GNN Extension
GCN, GraphSAGE, and GAT models for multi-label node classification.

All models follow the same interface:
  - Input:  x [num_nodes, in_dim], edge_index [2, num_edges]
  - Output: logits [num_nodes, num_classes]  (sigmoid applied in loss)

A learnable input embedding is used on top of the scalar degree feature,
since these datasets have no natural node features. This is a standard
technique for featureless graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class GCN(nn.Module):
    """2-layer Graph Convolutional Network (Kipf & Welling, 2017)."""

    def __init__(self, num_nodes, in_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        # Learnable node embedding (featureless graph technique)
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.lin_in = nn.Linear(in_dim, hidden_dim)

        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        node_ids = torch.arange(x.size(0), device=x.device)
        h = self.node_emb(node_ids) + self.lin_in(x)
        h = F.relu(self.conv1(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h  # raw logits


class GraphSAGE(nn.Module):
    """2-layer GraphSAGE (Hamilton et al., 2017)."""

    def __init__(self, num_nodes, in_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.lin_in = nn.Linear(in_dim, hidden_dim)

        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        node_ids = torch.arange(x.size(0), device=x.device)
        h = self.node_emb(node_ids) + self.lin_in(x)
        h = F.relu(self.conv1(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h


class GAT(nn.Module):
    """2-layer Graph Attention Network (Velickovic et al., 2018)."""

    def __init__(self, num_nodes, in_dim, hidden_dim, num_classes,
                 heads=8, dropout=0.5):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.lin_in = nn.Linear(in_dim, hidden_dim)

        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, num_classes, heads=1,
                             concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        node_ids = torch.arange(x.size(0), device=x.device)
        h = self.node_emb(node_ids) + self.lin_in(x)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.conv1(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h


def build_model(name, num_nodes, in_dim, hidden_dim, num_classes, **kwargs):
    """Factory: returns a model instance by name."""
    name = name.lower()
    if name == "gcn":
        return GCN(num_nodes, in_dim, hidden_dim, num_classes, **kwargs)
    if name == "graphsage":
        return GraphSAGE(num_nodes, in_dim, hidden_dim, num_classes, **kwargs)
    if name == "gat":
        return GAT(num_nodes, in_dim, hidden_dim, num_classes, **kwargs)
    raise ValueError(f"Unknown model: {name}")


if __name__ == "__main__":
    # Quick smoke test — build each model and run a forward pass
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from gnn.data_loader import build_pyg_data

    data = build_pyg_data("ppi")  # smallest dataset for fast test
    print(f"Testing on PPI: {data.num_nodes} nodes, {data.num_classes} classes\n")

    for model_name in ["gcn", "graphsage", "gat"]:
        model = build_model(
            model_name,
            num_nodes=data.num_nodes,
            in_dim=data.x.shape[1],
            hidden_dim=64,
            num_classes=data.num_classes,
        )
        out = model(data.x, data.edge_index)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {model_name.upper():<10} output: {tuple(out.shape)}  |  params: {n_params:,}")

    print("\nAll 3 models built and forward pass successful.")