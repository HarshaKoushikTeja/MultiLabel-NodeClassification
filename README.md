# Multi-Label Node Classification: Classical Embeddings vs Modern GNNs

> A rigorous benchmark of **7 graph learning methods** across **3 real-world datasets**, comparing classical random-walk embeddings (DeepWalk, Node2Vec) against modern Graph Neural Networks (GCN, GraphSAGE, GAT) — with multi-seed statistical validation.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.5-3C2179)
![License](https://img.shields.io/badge/License-MIT-green)

---

## TL;DR — Key Finding

**On featureless benchmark graphs, classical random-walk embeddings outperform modern GNNs.**

This is the headline insight: GCN, GraphSAGE, and GAT are architecturally designed to fuse *node features* with *graph structure*. When the datasets provide no node features (as is the case for BlogCatalog, PPI, and Wikipedia), these GNNs operate at a structural disadvantage — while DeepWalk and Node2Vec, purpose-built for featureless graphs, learn purely from connectivity and win.

This project demonstrates that **architectural sophistication does not guarantee better performance when input assumptions are violated** — a result established through proper 5-seed statistical validation, not a single lucky run.

---

## Project Origin

This work began as **P15**, a graduate course project (SML, Arizona State University) implementing DeepWalk and Node2Vec from scratch for multi-label node classification. It was then **extended into a full graph representation learning benchmark** by adding modern GNN baselines (GCN, GraphSAGE, GAT), multi-seed statistical validation, and embedding visualizations.

---

## Team & Contributions

This project was originally built by a 6-person team. The original P15 phase (datasets, DeepWalk, Node2Vec, classifiers, evaluation) was a collaborative effort:

| Member | Role | Contribution |
|--------|------|--------------|
| **Harsha Koushik Teja Aila** | Team Lead | Graph embedding interface (BaseEmbedding), integration pipeline, dataset verification, repository management, **GNN extension (GCN/GraphSAGE/GAT, multi-seed benchmark, t-SNE)** |
| **Prashant Rathod** | Data Engineer | Dataset acquisition (BlogCatalog, PPI, Wikipedia), graph construction, label processing, train/test splits |
| **Shaman Kanapathy** | DeepWalk Dev | DeepWalk from scratch — uniform random walk generator, skip-gram training, embedding output |
| **Priyanshu Gupta** | Node2Vec Dev | Node2Vec from scratch — biased walk generator, alias sampling, p/q tuning, combined embedding model |
| **Aditya Khurana** | ML Engineer | Baseline & embedding classifiers, One-vs-Rest Logistic Regression pipeline, combined experiment |
| **Sai Sagar Galli Raghu** | Eval & Viz | Evaluation metrics (Micro-F1, Macro-F1, Hamming Loss), result figures, full comparison tables |

> The **GNN extension** (everything in `src/gnn/` and `src/experiments/`) was developed as an individual continuation of the project.

---

## Headline Results

Micro-F1 across all 7 methods and 3 datasets. GNN values are **mean ± std over 5 seeds**; classical methods are single deterministic runs. Best per dataset in **bold**.

| Method | BlogCatalog | PPI | Wikipedia |
|--------|:-----------:|:---:|:---------:|
| Baseline (degree) | 0.1652 | 0.0937 | **0.3859** |
| DeepWalk | 0.2851 | 0.0882 | 0.3423 |
| Node2Vec | 0.2941 | 0.0932 | 0.3479 |
| **Combined (DW+N2V)** | **0.3298** | **0.1184** | 0.3744 |
| GCN | 0.1288 ± 0.001 | 0.0786 ± 0.003 | 0.3753 ± 0.001 |
| GraphSAGE | 0.1718 ± 0.003 | 0.1118 ± 0.003 | 0.3836 ± 0.001 |
| GAT | 0.1457 ± 0.035 | 0.1058 ± 0.001 | 0.2528 ± 0.151 |

**Observations:**
- The **Combined embedding** (DeepWalk + Node2Vec concatenated → 256-dim) is the strongest method on BlogCatalog and PPI.
- **GraphSAGE** is the best-performing GNN across all 3 datasets — its neighborhood sampling handles featureless graphs better than GCN or GAT.
- **GAT is unstable**: note the high variance (BlogCatalog ±0.035, Wikipedia ±0.151). Some seeds collapse entirely. This instability is only visible *because* of multi-seed evaluation — a single run would have hidden it.
- On **Wikipedia**, even the trivial degree baseline is competitive, indicating Part-of-Speech labels are more attribute-driven than structure-driven.

---

## Visualizations

### Classical vs GNN — Micro-F1 Comparison
![Comparison Chart](results/visualizations/classical_vs_gnn_comparison.png)

### Embedding Space (t-SNE) — DeepWalk vs Node2Vec
![t-SNE Grid](results/visualizations/tsne_comparison_grid.png)

The t-SNE projections honestly reflect the difficulty of these tasks: BlogCatalog shows partial cluster structure, while PPI and Wikipedia embeddings are diffuse — consistent with their low Micro-F1 scores.

---

## Datasets

| Dataset | Type | Nodes | Edges | Classes |
|---------|------|------:|------:|--------:|
| BlogCatalog | Social network | 10,312 | 333,983 | 39 |
| PPI | Protein interaction | 3,852 | 37,841 | 50 |
| Wikipedia | Word co-occurrence | 4,777 | 92,295 | 40 |

All datasets use an 80/20 stratified train/test split. These are the standard benchmarks from the original DeepWalk and Node2Vec papers, ensuring comparability with published work.

---

## Methods Implemented

**Classical (from scratch — no graph ML library):**
- **DeepWalk** — uniform random walks + Word2Vec skip-gram
- **Node2Vec** — biased random walks (p/q parameters) + alias sampling
- **Combined** — concatenation of DeepWalk + Node2Vec embeddings
- **Baseline** — node degree + One-vs-Rest Logistic Regression

**Modern GNNs (PyTorch Geometric):**
- **GCN** — Graph Convolutional Network (Kipf & Welling, 2017)
- **GraphSAGE** — neighborhood aggregation (Hamilton et al., 2017)
- **GAT** — Graph Attention Network (Veličković et al., 2018)

All GNNs use a learnable node embedding (a standard technique for featureless graphs), early stopping on a validation split, and a threshold sweep for multi-label prediction.

---

## Project Structure

```
MultiLabel-NodeClassification/
├── data/processed/          # graphs (.gpickle), labels, train/test splits
├── src/
│   ├── deepwalk/            # DeepWalk implementation         [Shaman]
│   ├── node2vec/            # Node2Vec implementation         [Priyanshu]
│   ├── classification/      # baseline + embedding classifiers [Aditya]
│   ├── embeddings/          # unified BaseEmbedding interface  [Harsha]
│   ├── evaluation/          # metrics (Micro/Macro-F1, Hamming) [Sagar]
│   ├── gnn/                 # GCN/GraphSAGE/GAT, trainer, t-SNE [GNN ext.-> Harsha]
│   └── experiments/         # multi-seed experiment runner      [GNN ext.-> Harsha]
├── results/
│   ├── gnn/                 # GNN results + combined comparison CSV
│   └── visualizations/      # t-SNE grids, comparison charts
├── reports/                 # final written report
└── requirements.txt
```

---

## How to Run

```bash
# 1. Clone and set up environment
git clone https://github.com/HarshaKoushikTeja/MultiLabel-NodeClassification.git
cd MultiLabel-NodeClassification
python -m venv venv
source venv/Scripts/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. (GPU) Install PyTorch + PyTorch Geometric with CUDA 12.1
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.5.3
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# 3. Run classical embeddings (DeepWalk / Node2Vec / baseline)
python src/deepwalk/run_deepwalk.py
python src/classification/classifier.py

# 4. Run the full GNN benchmark (3 models x 3 datasets x 5 seeds)
python src/experiments/run_gnn_experiments.py

# 5. Generate visualizations
python src/gnn/visualize_tsne.py
python src/gnn/plot_comparison.py
```

---

## Key Takeaways

1. **Negative results, interpreted well, are valuable.** GNNs underperforming here is not a failure — it is a precise demonstration of *when and why* GNNs need node features.
2. **Statistical rigor matters.** Multi-seed runs revealed GAT's training instability that a single run would have masked.
3. **The right tool depends on the data.** DeepWalk/Node2Vec dominate on featureless graphs; GNNs would likely win if node features were available (future work).

## Future Work

- Add Cora/Citeseer (feature-rich citation networks) to test the hypothesis that GNNs win when node features exist
- Hyperparameter tuning with Optuna for each model/dataset pair
- Ablation on GNN depth (over-smoothing analysis)
- Inductive evaluation (train on subgraph, test on unseen nodes)

---

## Tech Stack

`Python` · `PyTorch` · `PyTorch Geometric` · `NetworkX` · `scikit-learn` · `gensim` · `NumPy` · `Matplotlib` · `seaborn`

---

*Originally built as academic project P15 (SML, Arizona State University) and extended into a full graph representation learning benchmark.*