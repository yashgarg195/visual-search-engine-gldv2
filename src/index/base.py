"""Abstract Vector Index Interface.

Provides a unified interface for approximate and exact vector indices,
allowing future algorithms (e.g. HNSW, ScaNN) to be plugged in seamlessly.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple
import numpy as np


class BaseVectorIndex(ABC):
    """Abstract base class for all similarity search index implementations."""

    @abstractmethod
    def train(self, vectors: np.ndarray) -> None:
        """Train index structures (e.g., coarse quantizer and PQ codebooks)."""
        pass

    @abstractmethod
    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Add vectors with explicit int64 IDs to the index."""
        pass

    @abstractmethod
    def search(self, query_vectors: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search query vectors.

        Returns:
            scores: (N, top_k) array of inner product / similarity scores
            ids: (N, top_k) array of matched int64 FAISS IDs
        """
        pass

    @abstractmethod
    def save(self, index_path: Path) -> None:
        """Save index to disk."""
        pass

    @abstractmethod
    def load(self, index_path: Path) -> None:
        """Load index from disk."""
        pass

    @property
    @abstractmethod
    def total_vectors(self) -> int:
        """Return total number of vectors currently indexed."""
        pass

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """Return True if index is trained and ready for vector additions."""
        pass
