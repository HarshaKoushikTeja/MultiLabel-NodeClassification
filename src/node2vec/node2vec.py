"""
node2vec.py
───────────
Node2Vec implementation for the P15 Multi-Label Node Classification project.

Inherits from BaseEmbedding (src/embeddings/base_embedding.py).
Uses biased random walks (controlled by p and q) + Word2Vec skip-gram.

Key difference from DeepWalk:
    DeepWalk = Node2Vec with p=1, q=1 (unbiased walks)
    Node2Vec = biased walks via p (return) and q (in-out) parameters

Usage
-----
    from src.node2vec.node2vec import Node2Vec
    from src.embeddings.graph_loader import load_graph

    G     = load_graph('data/processed/blogcatalog.gpickle')
    model = Node2Vec(embedding_dim=128, walk_length=80, num_walks=10, p=1, q=1)
    model.fit(G)
    model.save('results/node2vec_blogcatalog.npy')
"""

import os
import random
import numpy as np
import networkx as nx
from gensim.models import Word2Vec

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.embeddings.base_embedding import BaseEmbedding


class Node2Vec(BaseEmbedding):
    """
    Node2Vec graph embedding model.

    Parameters
    ----------
    embedding_dim : int   — embedding vector size (default 128)
    walk_length   : int   — steps per random walk (default 80)
    num_walks     : int   — walks per node (default 10)
    p             : float — return parameter (default 1.0)
    q             : float — in-out parameter (default 1.0)
    window        : int   — Word2Vec context window (default 10)
    workers       : int   — parallel threads for Word2Vec (default 4)
    seed          : int   — random seed for reproducibility (default 42)
    """

    def __init__(
        self,
        embedding_dim: int   = 128,
        walk_length:   int   = 80,
        num_walks:     int   = 10,
        p:             float = 1.0,
        q:             float = 1.0,
        window:        int   = 10,
        workers:       int   = 4,
        seed:          int   = 42,
    ):
        super().__init__(embedding_dim=embedding_dim)
        self.walk_length = walk_length
        self.num_walks   = num_walks
        self.p           = p
        self.q           = q
        self.window      = window
        self.workers     = workers
        self.seed        = seed

        # Internal state
        self._graph        = None
        self._alias_nodes  = {}
        self._alias_edges  = {}
        self._word2vec     = None

    # ──────────────────────────────────────────────────────────────────
    # PUBLIC INTERFACE (BaseEmbedding contract)
    # ──────────────────────────────────────────────────────────────────

    def fit(self, graph: nx.Graph) -> None:
        """
        Learn Node2Vec embeddings from graph.
        Sets self.embeddings and self.node_to_idx after completion.

        Parameters
        ----------
        graph : nx.Graph — loaded via graph_loader.load_graph()
        """
        print(f"[Node2Vec] Starting fit on graph with "
              f"{graph.number_of_nodes()} nodes, "
              f"{graph.number_of_edges()} edges")
        print(f"[Node2Vec] p={self.p}, q={self.q}, "
              f"walks={self.num_walks}, walk_length={self.walk_length}, "
              f"dim={self.embedding_dim}")

        self._graph = graph
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Step 1: Precompute transition probabilities
        print("[Node2Vec] Precomputing transition probabilities...")
        self._precompute_transition_probs()

        # Step 2: Generate biased random walks
        print("[Node2Vec] Generating random walks...")
        walks = self._generate_walks()
        print(f"[Node2Vec] Total walks generated: {len(walks)}")

        # Step 3: Train Word2Vec skip-gram on walks
        print("[Node2Vec] Training Word2Vec...")
        self._word2vec = Word2Vec(
            sentences   = walks,
            vector_size = self.embedding_dim,
            window      = self.window,
            min_count   = 0,
            sg          = 1,          # skip-gram
            workers     = self.workers,
            seed        = self.seed,
            epochs      = 1,
        )

        # Step 4: Build embedding matrix + node_to_idx mapping
        # sorted node order ensures alignment with label matrix
        node_list = sorted(graph.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        self.embeddings  = np.array([
            self._word2vec.wv[str(node)] for node in node_list
        ])

        print(f"[Node2Vec] Done. Embedding matrix shape: {self.embeddings.shape}")

    def get_embedding(self, node_id) -> np.ndarray:
        """
        Return embedding vector for a single node.

        Parameters
        ----------
        node_id : int — node whose embedding you want

        Returns
        -------
        np.ndarray of shape [embedding_dim]
        """
        if self.embeddings is None:
            raise RuntimeError("Call fit() before get_embedding().")
        if node_id not in self.node_to_idx:
            raise KeyError(f"Node {node_id} not found in graph.")
        return self.embeddings[self.node_to_idx[node_id]]

    def save(self, filepath: str) -> None:
        """
        Save embedding matrix to .npy file.

        Parameters
        ----------
        filepath : str — e.g. 'results/node2vec_blogcatalog.npy'
        """
        if self.embeddings is None:
            raise RuntimeError("Call fit() before save().")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, self.embeddings)
        print(f"[Node2Vec] Embeddings saved → {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load embedding matrix from .npy file.

        Parameters
        ----------
        filepath : str — path to previously saved .npy file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Embedding file not found: {filepath}")
        self.embeddings = np.load(filepath)
        print(f"[Node2Vec] Embeddings loaded from {filepath} "
              f"— shape: {self.embeddings.shape}")

    # ──────────────────────────────────────────────────────────────────
    # PRIVATE METHODS — alias sampling and random walks
    # ──────────────────────────────────────────────────────────────────

    def _precompute_transition_probs(self):
        """Precompute alias tables for all nodes and edges."""
        G = self._graph

        # Alias table for each node (first step of walk)
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            weights   = [G[node][nbr].get('weight', 1.0) for nbr in neighbors]
            self._alias_nodes[node] = self._alias_setup(
                self._normalize(weights)
            )

        # Alias table for each edge (subsequent steps)
        for u, v in G.edges():
            self._alias_edges[(u, v)] = self._get_alias_edge(u, v)
            if not G.is_directed():
                self._alias_edges[(v, u)] = self._get_alias_edge(v, u)

    def _get_alias_edge(self, src, dst):
        """
        Compute biased transition probabilities for edge (src → dst).
        Applies p (return) and q (in-out) parameters.
        """
        G       = self._graph
        weights = []
        for nbr in G.neighbors(dst):
            w = G[dst][nbr].get('weight', 1.0)
            if nbr == src:
                weights.append(w / self.p)        # return to source
            elif G.has_edge(nbr, src):
                weights.append(w)                 # distance-1 from src
            else:
                weights.append(w / self.q)        # explore further
        return self._alias_setup(self._normalize(weights))

    def _generate_walks(self):
        """Generate all biased random walks from every node."""
        G     = self._graph
        nodes = list(G.nodes())
        walks = []

        for walk_num in range(self.num_walks):
            if (walk_num + 1) % 5 == 0:
                print(f"  Walk iteration {walk_num + 1}/{self.num_walks}")
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self._single_walk(node))

        return walks

    def _single_walk(self, start_node):
        """
        Perform one biased random walk starting from start_node.
        Returns list of node IDs as strings (for Word2Vec).
        """
        G    = self._graph
        walk = [start_node]

        while len(walk) < self.walk_length:
            cur       = walk[-1]
            neighbors = list(G.neighbors(cur))

            if not neighbors:
                break   # dead end — isolated or sink node

            if len(walk) == 1:
                # First step — use node alias (no previous node)
                J, q_arr = self._alias_nodes[cur]
                nxt      = neighbors[self._alias_draw(J, q_arr)]
            else:
                # Subsequent steps — use edge alias (biased by p and q)
                prev     = walk[-2]
                J, q_arr = self._alias_edges[(prev, cur)]
                nxt      = neighbors[self._alias_draw(J, q_arr)]

            walk.append(nxt)

        # Word2Vec expects strings
        return [str(n) for n in walk]

    # ──────────────────────────────────────────────────────────────────
    # ALIAS METHOD — O(1) sampling from arbitrary distributions
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(weights):
        """Normalize a list of weights to sum to 1."""
        total = sum(weights)
        return [w / total for w in weights] if total > 0 else weights

    @staticmethod
    def _alias_setup(probs):
        """
        Preprocess a probability distribution into alias table format.
        Enables O(1) sampling later via _alias_draw().

        Returns (J, q) where J[i] is the alias and q[i] is the threshold.
        """
        K       = len(probs)
        q_arr   = [0.0] * K
        J       = [0]   * K
        smaller = []
        larger  = []

        for i, prob in enumerate(probs):
            q_arr[i] = K * prob
            (smaller if q_arr[i] < 1.0 else larger).append(i)

        while smaller and larger:
            s, l    = smaller.pop(), larger.pop()
            J[s]    = l
            q_arr[l] = q_arr[l] + q_arr[s] - 1.0
            (smaller if q_arr[l] < 1.0 else larger).append(l)

        return J, q_arr

    @staticmethod
    def _alias_draw(J, q_arr):
        """Draw one sample in O(1) using precomputed alias table."""
        K = len(J)
        k = int(np.floor(np.random.random() * K))
        return k if np.random.random() < q_arr[k] else J[k]