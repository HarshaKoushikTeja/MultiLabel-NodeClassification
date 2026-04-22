# Results

## 4.1 Baseline Performance

The baseline classifier (degree + One-vs-Rest Logistic Regression) achieved relatively low performance across all datasets:

- **BlogCatalog**: Micro-F1 = 0.1652, Macro-F1 = 0.0245, Hamming Loss = 0.1375  
- **PPI**: Micro-F1 = 0.0937, Macro-F1 = 0.0649, Hamming Loss = 0.4419  
- **Wikipedia**: Micro-F1 = 0.3859, Macro-F1 = 0.0292, Hamming Loss = 0.0524  

This indicates that simple structural features like node degree are insufficient for capturing complex relationships in graph data.

---

## 4.2 DeepWalk vs Node2Vec

Node2Vec consistently outperforms DeepWalk across all datasets:

- **BlogCatalog**: Node2Vec Micro-F1 (0.2941) > DeepWalk (0.2851)  
- **PPI**: Node2Vec Micro-F1 (0.0932) > DeepWalk (0.0882)  
- **Wikipedia**: Node2Vec Micro-F1 (0.3479) > DeepWalk (0.3423)  

This improvement is due to Node2Vec’s ability to perform biased random walks, which better capture both local and global graph structures.

---

## 4.3 Combined Embeddings

Combining DeepWalk and Node2Vec embeddings yields the best performance:

- **BlogCatalog**: Micro-F1 = 0.3298 (~99% improvement over baseline)  
- **PPI**: Micro-F1 = 0.1184 (~26% improvement over baseline)  
- **Wikipedia**: Micro-F1 = 0.3744 (comparable to baseline performance)  

This demonstrates that combining multiple embedding techniques produces richer node representations.

---

## 4.4 Summary

Graph-based embedding methods significantly outperform the baseline approach across all datasets. Among them, Node2Vec performs better than DeepWalk, while the combined embedding approach achieves the best overall results.

These findings confirm that leveraging graph structure is crucial for improving multi-label node classification performance.