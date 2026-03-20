"""
base_embedding.py
─────────────────
Abstract base class for all graph embedding models in this project.

Every embedding implementation (DeepWalk, Node2Vec) MUST inherit from
this class and implement all abstract methods.

Usage (for Shaman / Priyanshu):
    from src.embeddings.base_embedding import BaseEmbedding

    class DeepWalk(BaseEmbedding):
        def fit(self, graph): ...
        def get_embedding(self, node_id): ...
        def save(self, filepath): ...
        def load(self, filepath): ...
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseEmbedding(ABC):
    """
    Abstract base class for graph node embedding models.

    All embedding models (DeepWalk, Node2Vec) must inherit this class
    and implement every abstract method below.

    Attributes
    ----------
    embedding_dim : int
        Dimensionality of the node embedding vectors. Default = 128.
    embeddings : np.ndarray or None
        Matrix of shape [num_nodes, embedding_dim] after fit() is called.
    node_to_idx : dict
        Maps node IDs (from the graph) to row indices in the embeddings matrix.
    """

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.embeddings: np.ndarray = None       # filled after fit()
        self.node_to_idx: dict = {}              # {node_id: matrix_row_index}

    # ──────────────────────────────────────────────────────────────────────────
    # ABSTRACT METHODS — every subclass MUST implement these
    # ──────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def fit(self, graph) -> None:
        """
        Learn node embeddings from the given graph.

        After this method runs, self.embeddings must be a numpy array
        of shape [num_nodes, embedding_dim], and self.node_to_idx must
        map every node in the graph to a valid row index.

        Parameters
        ----------
        graph : networkx.Graph
            The input graph loaded via graph_loader.py.

        Example
        -------
        model = DeepWalk(embedding_dim=128)
        model.fit(graph)
        # Now model.embeddings has shape [num_nodes, 128]
        """
        raise NotImplementedError

    @abstractmethod
    def get_embedding(self, node_id) -> np.ndarray:
        """
        Return the embedding vector for a single node.

        Parameters
        ----------
        node_id : int or str
            The node whose embedding you want.

        Returns
        -------
        np.ndarray
            1D array of shape [embedding_dim].

        Raises
        ------
        KeyError
            If node_id is not in the graph that was used to fit the model.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, filepath: str) -> None:
        """
        Save the learned embeddings to disk as a .npy file.

        The saved file must be a numpy array of shape [num_nodes, embedding_dim]
        where the row order matches self.node_to_idx.

        Parameters
        ----------
        filepath : str
            Path to save the .npy file.
            Convention: 'results/deepwalk_blogcatalog.npy'

        Example
        -------
        model.save('results/deepwalk_blogcatalog.npy')
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, filepath: str) -> None:
        """
        Load previously saved embeddings from a .npy file.

        Parameters
        ----------
        filepath : str
            Path to the .npy file to load.

        Example
        -------
        model = DeepWalk()
        model.load('results/deepwalk_blogcatalog.npy')
        """
        raise NotImplementedError

    # ──────────────────────────────────────────────────────────────────────────
    # SHARED UTILITY METHODS — already implemented, free to use in subclasses
    # ──────────────────────────────────────────────────────────────────────────

    def get_all_embeddings(self) -> np.ndarray:
        """
        Return the full embedding matrix of shape [num_nodes, embedding_dim].

        This is what the integration pipeline (pipeline.py) and the
        classifier (Aditya) will consume.

        Returns
        -------
        np.ndarray
            Shape [num_nodes, embedding_dim]. Rows correspond to nodes
            in the order defined by self.node_to_idx.

        Raises
        ------
        RuntimeError
            If fit() has not been called yet.
        """
        if self.embeddings is None:
            raise RuntimeError(
                "Embeddings not yet computed. Call fit(graph) first."
            )
        return self.embeddings

    def get_node_order(self) -> list:
        """
        Return the list of node IDs in the same order as the embedding rows.

        Useful for aligning embeddings with labels.

        Returns
        -------
        list
            Node IDs ordered so that node_order[i] corresponds to
            self.embeddings[i].
        """
        return sorted(self.node_to_idx, key=lambda n: self.node_to_idx[n])

    def embedding_shape(self) -> tuple:
        """
        Return the shape of the embedding matrix.

        Returns
        -------
        tuple
            (num_nodes, embedding_dim) or None if not yet fitted.
        """
        if self.embeddings is None:
            return None
        return self.embeddings.shape

    def __repr__(self) -> str:
        status = (
            f"fitted — shape {self.embeddings.shape}"
            if self.embeddings is not None
            else "not fitted"
        )
        return f"{self.__class__.__name__}(dim={self.embedding_dim}, {status})"
