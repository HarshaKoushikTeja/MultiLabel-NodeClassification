"""
measure_timing.py — GNN Extension
Measures training time and parameter count for each GNN model on each dataset.
Runs 1 seed per (model, dataset) — representative for an efficiency table.
"""

import os
import sys
import time
import json
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gnn.data_loader import build_pyg_data
from gnn.models import build_model
from gnn.trainer import (
    set_seed, make_val_split, best_threshold, evaluate, DEVICE
)


DATASETS = ["blogcatalog", "ppi", "wikipedia"]
MODELS = ["gcn", "graphsage", "gat"]

CONFIG = {
    "blogcatalog": {"hidden_dim": 128, "lr": 0.01, "max_epochs": 300, "patience": 40},
    "ppi":         {"hidden_dim": 128, "lr": 0.01, "max_epochs": 300, "patience": 40},
    "wikipedia":   {"hidden_dim": 128, "lr": 0.01, "max_epochs": 300, "patience": 40},
}


def timed_run(dataset_name, model_name, cfg, seed=0):
    set_seed(seed)
    data = build_pyg_data(dataset_name).to(DEVICE)
    train_mask, val_mask = make_val_split(data, seed=seed)
    train_mask = train_mask.to(DEVICE)
    val_mask = val_mask.to(DEVICE)

    model = build_model(
        model_name,
        num_nodes=data.num_nodes,
        in_dim=data.x.shape[1],
        hidden_dim=cfg["hidden_dim"],
        num_classes=data.num_classes,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                 weight_decay=5e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Time the full training loop (with early stopping, like real runs)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    actual_epochs = 0

    for epoch in range(1, cfg["max_epochs"] + 1):
        actual_epochs = epoch
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_loss = criterion(val_out[val_mask], data.y[val_mask]).item()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= cfg["patience"]:
            break

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    train_time = time.perf_counter() - t0

    # GPU memory used (if CUDA)
    gpu_mem_mb = 0
    if DEVICE.type == "cuda":
        gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats()

    return {
        "dataset": dataset_name,
        "model": model_name,
        "params": n_params,
        "epochs_run": actual_epochs,
        "train_time_sec": round(train_time, 2),
        "gpu_mem_mb": round(gpu_mem_mb, 1),
    }


def main():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    results = []
    for dataset in DATASETS:
        cfg = CONFIG[dataset]
        for model in MODELS:
            print(f"Timing {model.upper()} on {dataset.upper()}...")
            r = timed_run(dataset, model, cfg)
            results.append(r)
            print(f"  params={r['params']:,}  "
                  f"epochs={r['epochs_run']}  "
                  f"time={r['train_time_sec']}s  "
                  f"gpu_mem={r['gpu_mem_mb']}MB")

    os.makedirs("results/gnn", exist_ok=True)
    with open("results/gnn/timing_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*55}")
    print("TIMING SUMMARY")
    print('='*55)
    print(f"{'Dataset':<14}{'Model':<12}{'Params':>10}"
          f"{'Epochs':>8}{'Time(s)':>10}{'GPU(MB)':>10}")
    print("-" * 64)
    for r in results:
        print(f"{r['dataset']:<14}{r['model']:<12}{r['params']:>10,}"
              f"{r['epochs_run']:>8}{r['train_time_sec']:>10}"
              f"{r['gpu_mem_mb']:>10}")

    print("\nSaved -> results/gnn/timing_results.json")


if __name__ == "__main__":
    main()