"""FAISS HNSW-PQ Index Implementation with Hierarchical Navigable Small World Graph & Product Quantization."""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import faiss
import numpy as np
import pandas as pd

from src.index.base import BaseVectorIndex
from src.utils.io import atomic_save_json, load_json
from src.utils.logging import get_logger

logger = get_logger("hnsw_pq_index")


def validate_hnsw_pq_parameters(d: int, m: int, M: int, num_training_vectors: int) -> None:
    """Validate HNSW-PQ parameters against FAISS constraints and mathematical rules."""
    if d % m != 0:
        raise ValueError(
            f"Embedding dimension ({d}) must be divisible by m ({m}). "
            f"Valid options for d={d}: {[x for x in [8, 16, 32, 64, 128] if d % x == 0]}"
        )

    if M < 4 or M > 128:
        raise ValueError(f"HNSW links per node (M) must be between 4 and 128, got {M}")

    # PQ requires at least 256 training points (for 8-bit sub-quantizers)
    if num_training_vectors < 256:
        raise ValueError(
            f"Training vector count ({num_training_vectors}) is smaller than 256. "
            f"PQ codebook training requires at least 256 points."
        )


class HNSWPQIndex(BaseVectorIndex):
    """FAISS HNSW-PQ Index using Inner Product similarity, Hierarchical Navigable Graph, and IndexIDMap2."""

    def __init__(
        self,
        d: int = 512,
        m: int = 64,
        M: int = 32,
        ef_construction: int = 64,
        ef_search: int = 64,
    ):
        self.d = d
        self.m = m
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.metric = faiss.METRIC_INNER_PRODUCT

        # Internal FAISS index: IDMap2 wrapping IndexHNSWPQ
        self._index: Optional[faiss.IndexIDMap2] = None

    def build_empty(self) -> None:
        """Instantiate an empty, untrained IDMap2(HNSWPQ) index."""
        factory_str = f"IDMap2,HNSW{self.M},PQ{self.m}"
        logger.debug("Constructing FAISS index with factory: %s", factory_str)
        self._index = faiss.index_factory(self.d, factory_str, self.metric)

        # Configure efConstruction on the underlying HNSW
        sub = faiss.downcast_index(self._index.index)
        sub.thisown = False
        sub.hnsw.efConstruction = self.ef_construction
        sub.hnsw.efSearch = self.ef_search

    def set_ef_search(self, ef_search: int) -> None:
        """Adjust runtime HNSW search exploration depth."""
        if ef_search < 1:
            raise ValueError(f"ef_search must be >= 1, got {ef_search}")
        self.ef_search = ef_search
        if self._index is not None and hasattr(self._index, "index"):
            sub = faiss.downcast_index(self._index.index)
            sub.thisown = False
            if hasattr(sub, "hnsw"):
                sub.hnsw.efSearch = ef_search
            logger.debug("Updated HNSW efSearch to %d", ef_search)

    def train(self, vectors: np.ndarray) -> None:
        """Train PQ codebooks using representative float32 vectors."""
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        validate_hnsw_pq_parameters(
            d=self.d,
            m=self.m,
            M=self.M,
            num_training_vectors=len(vectors),
        )

        if self._index is None:
            self.build_empty()

        logger.info(
            "Training HNSW-PQ codebooks (d=%d, m=%d, M=%d, ef_c=%d) on %d vectors...",
            self.d,
            self.m,
            self.M,
            self.ef_construction,
            len(vectors),
        )
        t_start = time.perf_counter()

        # On macOS ARM64, multi-threaded k-means in faiss-cpu can crash; protect with single-thread training
        old_threads = faiss.omp_get_max_threads() if hasattr(faiss, "omp_get_max_threads") else 1
        try:
            if hasattr(faiss, "omp_set_num_threads"):
                faiss.omp_set_num_threads(1)
            self._index.train(vectors)
        finally:
            if hasattr(faiss, "omp_set_num_threads"):
                faiss.omp_set_num_threads(old_threads)

        t_elapsed = time.perf_counter() - t_start
        logger.info("HNSW-PQ training completed in %.2f seconds.", t_elapsed)

    def add(self, vectors: np.ndarray, ids: np.ndarray, num_threads: int = 4) -> None:
        """Add vectors with explicit 64-bit integer IDs into the HNSW graph."""
        if self._index is None or not self.is_trained:
            raise RuntimeError("Index must be built and trained before adding vectors.")

        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        if ids.dtype != np.int64:
            ids = ids.astype(np.int64)

        if len(vectors) != len(ids):
            raise ValueError(f"Number of vectors ({len(vectors)}) != number of IDs ({len(ids)})")

        t_start = time.perf_counter()
        old_threads = faiss.omp_get_max_threads() if hasattr(faiss, "omp_get_max_threads") else 1
        try:
            if hasattr(faiss, "omp_set_num_threads"):
                faiss.omp_set_num_threads(max(1, num_threads))
            self._index.add_with_ids(vectors, ids)
        finally:
            if hasattr(faiss, "omp_set_num_threads"):
                faiss.omp_set_num_threads(old_threads)
        t_elapsed = time.perf_counter() - t_start
        logger.debug(
            "Added %d vectors into HNSW-PQ in %.2f seconds (total: %d)",
            len(vectors),
            t_elapsed,
            self.total_vectors,
        )

    def search(self, query_vectors: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Query nearest neighbors via HNSW graph traversal and PQ distance estimation."""
        if self._index is None:
            raise RuntimeError("Index has not been initialized.")

        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype(np.float32)

        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)

        # Ensure current ef_search is applied
        if hasattr(self._index, "index"):
            sub = faiss.downcast_index(self._index.index)
            sub.thisown = False
            if hasattr(sub, "hnsw"):
                sub.hnsw.efSearch = self.ef_search

        old_threads = faiss.omp_get_max_threads() if hasattr(faiss, "omp_get_max_threads") else 1
        try:
            if hasattr(faiss, "omp_set_num_threads"):
                faiss.omp_set_num_threads(1)
            scores, ids = self._index.search(query_vectors, top_k)
        finally:
            if hasattr(faiss, "omp_set_num_threads"):
                faiss.omp_set_num_threads(old_threads)

        return scores, ids

    def reset(self) -> None:
        """Clear all vectors and reset graph structure while preserving trained codebooks."""
        if self._index is not None:
            self._index.reset()

    def save(self, index_path: Path) -> None:
        """Persist HNSW-PQ index and metadata to disk."""
        if self._index is None:
            raise RuntimeError("Cannot save an uninitialized index.")

        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Writing HNSW-PQ index to %s...", index_path)
        faiss.write_index(self._index, str(index_path))

        # Write companion metadata
        meta = {
            "type": "hnsw_pq",
            "d": self.d,
            "m": self.m,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "metric": "inner_product",
            "total_vectors": self.total_vectors,
            "is_trained": self.is_trained,
        }
        meta_path = index_path.with_suffix(".meta.json")
        atomic_save_json(meta, meta_path)
        logger.info("Saved HNSW-PQ index metadata to %s", meta_path)

    def load(self, index_path: Path) -> None:
        """Load a persisted HNSW-PQ index from disk."""
        index_path = Path(index_path)
        if not index_path.is_file():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        logger.info("Loading HNSW-PQ index from %s", index_path)
        raw_index = faiss.read_index(str(index_path))
        self._index = raw_index

        # Read companion metadata if available
        meta_path = index_path.with_suffix(".meta.json")
        if meta_path.is_file():
            try:
                meta = load_json(meta_path)
                self.d = meta.get("d", self.d)
                self.m = meta.get("m", self.m)
                self.M = meta.get("M", self.M)
                self.ef_construction = meta.get("ef_construction", self.ef_construction)
                self.ef_search = meta.get("ef_search", self.ef_search)
            except Exception as e:
                logger.warning("Failed reading companion meta %s: %s", meta_path, e)

        # Set default efSearch
        if self._index is not None and hasattr(self._index, "index"):
            sub = faiss.downcast_index(self._index.index)
            sub.thisown = False
            if hasattr(sub, "hnsw"):
                sub.hnsw.efSearch = self.ef_search

        logger.info("Loaded HNSW-PQ index with %d vectors (efSearch=%d).", self.total_vectors, self.ef_search)

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index is not None else 0

    @property
    def is_trained(self) -> bool:
        return self._index.is_trained if self._index is not None else False
