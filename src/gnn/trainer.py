"""
trainer.py — GNN Extension
Trains a GNN model for multi-label node classification with:
  - Proper train/val/test split (val carved from train for early stopping)
  - Early stopping on validation loss
  - Multi-seed runs for statistical rigor (mean ± std)
  - GPU support

Metrics: Micro-F1, Macro-F1, Hamming Loss (same as DeepWalk/Node2Vec).
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, hamming_loss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gnn.data_loader import build_pyg_data
from gnn.models import build_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_val_split(data, val_frac=0.15, seed=42):
    """Carve a validation set out of the training mask for early stopping."""
    rng = np.random.RandomState(seed)
    train_idx = data.train_mask.nonzero(as_tuple=True)[0].cpu().numpy()
    rng.shuffle(train_idx)
    n_val = int(len(train_idx) * val_frac)
    val_idx = train_idx[:n_val]
    new_train_idx = train_idx[n_val:]

    train_mask = torch.zeros_like(data.train_mask)
    val_mask = torch.zeros_like(data.train_mask)
    train_mask[new_train_idx] = True
    val_mask[val_idx] = True
    return train_mask, val_mask


@torch.no_grad()
def evaluate(model, data, mask, threshold=0.5):
    """Return Micro-F1, Macro-F1, Hamming Loss on masked nodes at given threshold."""
    model.eval()
    logits = model(data.x, data.edge_index)
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    y_true = data.y[mask].cpu().numpy()
    y_pred = preds[mask].cpu().numpy()

    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    hamm = hamming_loss(y_true, y_pred)
    return micro, macro, hamm


@torch.no_grad()
def best_threshold(model, data, val_mask):
    """Sweep thresholds on validation set, return the one maximizing Micro-F1."""
    model.eval()
    logits = model(data.x, data.edge_index)
    probs = torch.sigmoid(logits)
    y_true = data.y[val_mask].cpu().numpy()
    val_probs = probs[val_mask].cpu().numpy()

    best_t, best_micro = 0.5, 0.0
    for t in np.arange(0.05, 0.51, 0.05):
        y_pred = (val_probs > t).astype(float)
        micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
        if micro > best_micro:
            best_micro = micro
            best_t = float(round(t, 2))
    return best_t


def train_one_run(dataset_name, model_name, seed,
                   hidden_dim=128, lr=0.01, weight_decay=5e-4,
                   max_epochs=300, patience=30, verbose=False):
    """Train one model with one seed. Returns test metrics at best val epoch."""
    set_seed(seed)
    data = build_pyg_data(dataset_name).to(DEVICE)
    train_mask, val_mask = make_val_split(data, seed=seed)
    train_mask = train_mask.to(DEVICE)
    val_mask = val_mask.to(DEVICE)

    model = build_model(
        model_name,
        num_nodes=data.num_nodes,
        in_dim=data.x.shape[1],
        hidden_dim=hidden_dim,
        num_classes=data.num_classes,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_test_metrics = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        # Validation loss for early stopping
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_loss = criterion(val_out[val_mask], data.y[val_mask]).item()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            t = best_threshold(model, data, val_mask)
            best_test_metrics = evaluate(model, data, data.test_mask, threshold=t)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and epoch % 20 == 0:
            print(f"    epoch {epoch:3d} | train_loss {loss.item():.4f} "
                  f"| val_loss {val_loss:.4f}")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"    early stop at epoch {epoch}")
            break

    return best_test_metrics  # (micro, macro, hamming)


def run_multi_seed(dataset_name, model_name, seeds=(0, 1, 2, 3, 4), **kwargs):
    """Run multiple seeds and return mean ± std for each metric."""
    print(f"\n{'='*55}")
    print(f"{model_name.upper()} on {dataset_name.upper()} "
          f"({len(seeds)} seeds, device={DEVICE})")
    print('='*55)

    micros, macros, hamms = [], [], []
    for seed in seeds:
        micro, macro, hamm = train_one_run(
            dataset_name, model_name, seed, **kwargs
        )
        micros.append(micro)
        macros.append(macro)
        hamms.append(hamm)
        print(f"  seed {seed}: Micro-F1={micro:.4f}  "
              f"Macro-F1={macro:.4f}  Hamming={hamm:.4f}")

    result = {
        "dataset": dataset_name,
        "model": model_name,
        "seeds": list(seeds),
        "micro_f1_mean": float(np.mean(micros)),
        "micro_f1_std":  float(np.std(micros)),
        "macro_f1_mean": float(np.mean(macros)),
        "macro_f1_std":  float(np.std(macros)),
        "hamming_mean":  float(np.mean(hamms)),
        "hamming_std":   float(np.std(hamms)),
    }
    print(f"\n  RESULT: Micro-F1 = {result['micro_f1_mean']:.4f} "
          f"± {result['micro_f1_std']:.4f}")
    print(f"          Macro-F1 = {result['macro_f1_mean']:.4f} "
          f"± {result['macro_f1_std']:.4f}")
    print(f"          Hamming  = {result['hamming_mean']:.4f} "
          f"± {result['hamming_std']:.4f}")
    return result


if __name__ == "__main__":
    # Smoke test: 2 seeds, GCN on PPI (fast)
    result = run_multi_seed("ppi", "gcn", seeds=(0, 1),
                            hidden_dim=64, max_epochs=100, patience=20,
                            verbose=True)
    print("\nSmoke test complete.")