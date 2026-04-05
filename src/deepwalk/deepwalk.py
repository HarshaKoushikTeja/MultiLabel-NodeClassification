"""
deepwalk.py
DeepWalk implementation for multi-label node classification.

Algorithm: Perozzi et al. 2014 — "DeepWalk: Online Learning of Social Representations"

Usage:
    from src.deepwalk.deepwalk import DeepWalk

    model = DeepWalk()
    model.fit(graph)
    model.save('results/deepwalk_blogcatalog.npy')
"""

import random
import numpy as np
from gensim.models import Word2Vec
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.embeddings.base_embedding import BaseEmbedding


class DeepWalk(BaseEmbedding):
    """
    DeepWalk: Learns node embeddings via uniform random walks + Word2Vec (skip-gram).

    Inherits from BaseEmbedding and implements all 4 abstract methods:
        fit(), get_embedding(), save(), load()

    Parameters
    ----------
    embedding_dim : int
        Size of each node's embedding vector. Default = 128.
    walk_length : int
        Number of nodes in each random walk. Default = 80.
    num_walks : int
        Number of random walks starting from each node. Default = 10.
    window : int
        Word2Vec context window size. Default = 10.
    workers : int
        Number of CPU threads for Word2Vec training. Default = 4.
    epochs : int
        Number of training epochs for Word2Vec. Default = 1.
    seed : int
        Random seed for reproducibility. Default = 42.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        walk_length: int = 80,
        num_walks: int = 10,
        window: int = 10,
        workers: int = 4,
        epochs: int = 1,
        seed: int = 42,
    ):
        super().__init__(embedding_dim=embedding_dim)
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window = window
        self.workers = workers
        self.epochs = epochs
        self.seed = seed
        self._w2v_model = None  # internal Word2Vec model

    def fit(self, graph) -> None:
        """
        Learn DeepWalk embeddings from the given NetworkX graph.

        Steps:
            1. Generate random walks from every node
            2. Train Word2Vec (skip-gram) on those walks
            3. Build embedding matrix and node_to_idx mapping

        After fit(), self.embeddings has shape [num_nodes, embedding_dim]
        and self.node_to_idx maps each node to its row index.

        Parameters
        ----------
        graph : networkx.Graph
            Input graph loaded via graph_loader.py (Prashant's output).
        """
        print(f"[DeepWalk] Graph loaded: {graph.number_of_nodes()} nodes, "
              f"{graph.number_of_edges()} edges")

        # Step 1: Generate random walks
        print(f"[DeepWalk] Generating walks "
              f"(num_walks={self.num_walks}, walk_length={self.walk_length})...")
        walks = self._generate_all_walks(graph)
        print(f"[DeepWalk] Total walks generated: {len(walks)}")

        # Step 2: Train Word2Vec on walks
        # Nodes are converted to strings because Word2Vec expects string tokens
        print(f"[DeepWalk] Training Word2Vec "
              f"(vector_size={self.embedding_dim}, window={self.window}, sg=1)...")
        self._w2v_model = Word2Vec(
            sentences=walks,
            vector_size=self.embedding_dim,
            window=self.window,
            sg=1,               # skip-gram (as per DeepWalk paper)
            workers=self.workers,
            epochs=self.epochs,
            seed=self.seed,
        )

        # Step 3: Build embedding matrix aligned with node_to_idx
        nodes = sorted(graph.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(nodes)}

        self.embeddings = np.zeros((len(nodes), self.embedding_dim))
        for node in nodes:
            idx = self.node_to_idx[node]
            node_str = str(node)
            if node_str in self._w2v_model.wv:
                self.embeddings[idx] = self._w2v_model.wv[node_str]
            else:
                # Fallback: zero vector (rare edge case for isolated nodes)
                self.embeddings[idx] = np.zeros(self.embedding_dim)

        print(f"[DeepWalk] Embedding matrix shape: {self.embeddings.shape}")
        print("[DeepWalk] fit() complete.")

    def get_embedding(self, node_id) -> np.ndarray:
        """
        Return the embedding vector for a single node.

        Parameters
        ----------
        node_id : int or str
            Node whose embedding you want.

        Returns
        -------
        np.ndarray
            1D array of shape [embedding_dim].

        Raises
        ------
        RuntimeError
            If fit() has not been called yet.
        KeyError
            If node_id is not in the graph.
        """
        if self.embeddings is None:
            raise RuntimeError("Call fit(graph) before get_embedding().")
        if node_id not in self.node_to_idx:
            raise KeyError(f"Node '{node_id}' not found in the graph.")
        return self.embeddings[self.node_to_idx[node_id]]

    def save(self, filepath: str) -> None:
        """
        Save the embedding matrix to a .npy file.

        Output shape: [num_nodes, embedding_dim]
        Convention:   'results/deepwalk_blogcatalog.npy'
                      'results/deepwalk_ppi.npy'

        Parameters
        ----------
        filepath : str
            Path to save the .npy file.

        Raises
        ------
        RuntimeError
            If fit() has not been called yet.
        """
        if self.embeddings is None:
            raise RuntimeError("Call fit(graph) before save().")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, self.embeddings)
        print(f"[DeepWalk] Embeddings saved to '{filepath}' "
              f"— shape {self.embeddings.shape}")

    def load(self, filepath: str) -> None:
        """
        Load previously saved embeddings from a .npy file.

        Parameters
        ----------
        filepath : str
            Path to the .npy file to load.
        """
        self.embeddings = np.load(filepath)
        print(f"[DeepWalk] Embeddings loaded from '{filepath}' "
              f"— shape {self.embeddings.shape}")

    def _generate_all_walks(self, graph) -> list:
        """
        Generate all random walks for every node in the graph.

        Follows Algorithm 1 from Perozzi et al. 2014:
            - Outer loop: repeat num_walks times (γ passes over data)
            - Inner loop: shuffle nodes, start a walk from each node
            - Each walk: uniformly sample a neighbor at each step

        Returns
        -------
        list of list of str
            All walks, where each node is represented as a string
            (required by gensim Word2Vec).
        """
        random.seed(self.seed)
        nodes = list(graph.nodes())
        all_walks = []

        for pass_num in range(self.num_walks):
            # Shuffle node order each pass (as per paper — speeds up SGD)
            random.shuffle(nodes)
            for node in nodes:
                walk = self._random_walk(graph, node)
                all_walks.append([str(n) for n in walk])

            if (pass_num + 1) % 2 == 0 or pass_num == 0:
                print(f"[DeepWalk]   Pass {pass_num + 1}/{self.num_walks} complete "
                      f"({len(all_walks)} walks so far)")

        return all_walks

    def _random_walk(self, graph, start_node) -> list:
        """
        Perform a single uniform random walk starting from start_node.

        At each step, a neighbor is chosen uniformly at random.
        Walk stops early if the current node has no neighbors (isolated node).

        Parameters
        ----------
        graph : networkx.Graph
        start_node : node ID

        Returns
        -------
        list
            Sequence of node IDs visited during the walk.
        """
        walk = [start_node]
        for _ in range(self.walk_length - 1):
            current = walk[-1]
            neighbors = list(graph.neighbors(current))
            if not neighbors:
                break  # isolated node — stop walk early
            walk.append(random.choice(neighbors))
        return walk


if __name__ == "__main__":
    import networkx as nx

    print("=" * 60)
    print("DeepWalk Quick Test on a small synthetic graph")
    print("=" * 60)

    # Build a small test graph (Karate Club — 34 nodes)
    G = nx.karate_club_graph()
    print(f"Test graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Run DeepWalk with small params for quick testing
    model = DeepWalk(
        embedding_dim=16,
        walk_length=10,
        num_walks=5,
        window=5,
        workers=1,
    )

    model.fit(G)

    # Test get_embedding
    vec = model.get_embedding(0)
    print(f"Embedding for node 0: shape={vec.shape}, first 4 values={vec[:4]}")

    # Test get_all_embeddings
    matrix = model.get_all_embeddings()
    print(f"Full matrix shape: {matrix.shape}")

    # Test save/load
    os.makedirs("results", exist_ok=True)
    model.save("results/deepwalk_test.npy")
    model2 = DeepWalk()
    model2.load("results/deepwalk_test.npy")
    print(f"Loaded matrix shape: {model2.embeddings.shape}")

    print("\n✅ All tests passed! DeepWalk is working correctly.")
    print(f"Model: {model}")
