"""Exact Baseline Index using IndexFlatIP with IndexIDMap2."""

import os
from pathlib import Path
from typing import Tuple
import faiss
import numpy as np

from src.index.base import BaseVectorIndex
from src.utils.logging import get_logger

logger = get_logger("flat_index")


class FlatIPIndex(BaseVectorIndex):
    """Exact cosine/inner-product similarity search index using FAISS IndexFlatIP."""

    def __init__(self, d: int = 512):
        self.d = d
        self._flat = faiss.IndexFlatIP(self.d)
        self._index = faiss.IndexIDMap2(self._flat)

    @property
    def is_trained(self) -> bool:
        return True  # Flat index does not require training

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal

    def train(self, vectors: np.ndarray) -> None:
        """Flat index requires no training."""
        pass

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Add vectors with explicit 64-bit integer IDs."""
        if len(vectors) != len(ids):
            raise ValueError(f"Vectors count ({len(vectors)}) != IDs count ({len(ids)})")

        import platform
        if platform.system() == "Darwin" and hasattr(faiss, "omp_set_num_threads"):
            faiss.omp_set_num_threads(1)

        vectors_f32 = np.ascontiguousarray(vectors, dtype=np.float32)
        ids_i64 = np.ascontiguousarray(ids, dtype=np.int64)
        self._index.add_with_ids(vectors_f32, ids_i64)

    def search(self, query_vectors: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Perform exact exhaustive search."""
        if self._index.ntotal == 0:
            return np.empty((len(query_vectors), 0), dtype=np.float32), np.empty(
                (len(query_vectors), 0), dtype=np.int64
            )

        import platform
        if platform.system() == "Darwin" and hasattr(faiss, "omp_set_num_threads"):
            faiss.omp_set_num_threads(1)

        queries_f32 = np.ascontiguousarray(query_vectors, dtype=np.float32)
        effective_k = min(top_k, self._index.ntotal)
        scores, ids = self._index.search(queries_f32, effective_k)
        return scores, ids

    def save(self, index_path: Path) -> None:
        """Save exact index to disk."""
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = index_path.with_suffix(f"{index_path.suffix}.tmp.{os.getpid()}")
        try:
            faiss.write_index(self._index, str(temp_path))
            os.replace(temp_path, index_path)
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def load(self, index_path: Path) -> None:
        """Load exact index from disk."""
        index_path = Path(index_path)
        if not index_path.is_file():
            raise FileNotFoundError(f"Flat index file not found: {index_path}")
        raw = faiss.read_index(str(index_path))
        self._index = raw
        self.d = self._index.d
