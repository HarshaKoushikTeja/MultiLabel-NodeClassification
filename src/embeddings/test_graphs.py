"""
test_graphs.py
──────────────
Quick verification script to confirm all 3 datasets load correctly.
Run this once after pulling main to verify your local setup is working.

Usage:
    python src/embeddings/test_graphs.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from graph_loader import load_graph, load_labels, graph_stats

DATASETS = [
    {
        "name":        "BlogCatalog",
        "graph_path":  "data/processed/blogcatalog.gpickle",
        "labels_path": "data/processed/labels_blogcatalog.csv",
    },
    {
        "name":        "PPI",
        "graph_path":  "data/processed/ppi.gpickle",
        "labels_path": "data/processed/labels_ppi.csv",
    },
    {
        "name":        "Wikipedia",
        "graph_path":  "data/processed/wikipedia.gpickle",
        "labels_path": "data/processed/labels_wikipedia.csv",
    },
]

def test_all_datasets():
    results = []

    for ds in DATASETS:
        print(f"\n{'='*55}")
        print(f"  Testing: {ds['name']}")
        print(f"{'='*55}")

        try:
            graph  = load_graph(ds["graph_path"])
            stats  = graph_stats(graph)
            labels = load_labels(ds["labels_path"])

            # Basic sanity checks
            assert graph.number_of_nodes() > 0,  "Graph has no nodes!"
            assert graph.number_of_edges() > 0,  "Graph has no edges!"
            assert len(labels) > 0,              "Labels are empty!"
            assert len(labels) == graph.number_of_nodes(), \
                f"Node count mismatch: graph={graph.number_of_nodes()}, labels={len(labels)}"

            num_classes = len(next(iter(labels.values())))
            print(f"  classes       : {num_classes}")
            print(f"  PASSED ✓")

            results.append({
                "dataset":   ds["name"],
                "status":    "PASSED",
                "nodes":     graph.number_of_nodes(),
                "edges":     graph.number_of_edges(),
                "classes":   num_classes,
            })

        except FileNotFoundError as e:
            print(f"  SKIPPED — file not found: {e}")
            results.append({"dataset": ds["name"], "status": "SKIPPED"})

        except AssertionError as e:
            print(f"  FAILED — {e}")
            results.append({"dataset": ds["name"], "status": "FAILED", "error": str(e)})

        except Exception as e:
            print(f"  ERROR — {e}")
            results.append({"dataset": ds["name"], "status": "ERROR", "error": str(e)})

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Dataset':<15} {'Status':<10} {'Nodes':<8} {'Edges':<10} {'Classes'}")
    print(f"  {'-'*50}")
    for r in results:
        if r["status"] == "PASSED":
            print(f"  {r['dataset']:<15} {r['status']:<10} {r['nodes']:<8} {r['edges']:<10} {r['classes']}")
        else:
            print(f"  {r['dataset']:<15} {r['status']}")

    passed  = sum(1 for r in results if r["status"] == "PASSED")
    skipped = sum(1 for r in results if r["status"] == "SKIPPED")
    failed  = sum(1 for r in results if r["status"] in ("FAILED", "ERROR"))

    print(f"\n  {passed} passed  |  {skipped} skipped  |  {failed} failed")
    print(f"{'='*55}\n")

    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    test_all_datasets()