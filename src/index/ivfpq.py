"""FAISS IVF-PQ Index Implementation with Explicit ID Mapping and Parameter Validation."""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import faiss
import numpy as np
import pandas as pd

from src.index.base import BaseVectorIndex
from src.index.threading_utils import faiss_single_thread_on_macos
from src.utils.io import atomic_save_json, atomic_save_parquet, load_json
from src.utils.logging import get_logger

logger = get_logger("ivfpq_index")


def validate_ivfpq_parameters(d: int, nlist: int, m: int, nbits: int, num_training_vectors: int) -> None:
    """Validate IVF-PQ parameters against FAISS constraints and mathematical rules."""
    # 1. Dimension divisibility check
    if d % m != 0:
        raise ValueError(
            f"Embedding dimension ({d}) must be divisible by m ({m}). "
            f"Valid options for d={d}: {[x for x in [8, 16, 32, 64, 128] if d % x == 0]}"
        )

    # 2. Minimum centroids check
    if nlist < 1:
        raise ValueError(f"nlist must be >= 1, got {nlist}")

    # 3. Training points vs nlist
    if num_training_vectors < nlist:
        safe_nlist = max(1, 2 ** int(np.floor(np.log2(max(1, num_training_vectors // 30)))))
        raise ValueError(
            f"Training vector count ({num_training_vectors}) is smaller than nlist ({nlist}). "
            f"FAISS k-means clustering requires at least nlist training points. "
            f"Recommended safe nlist for {num_training_vectors} vectors: {safe_nlist} (or lower)."
        )

    # 4. PQ subquantizer centroid count vs training size
    min_pq_points = 2**nbits
    if num_training_vectors < min_pq_points:
        raise ValueError(
            f"Training vector count ({num_training_vectors}) is smaller than 2^nbits ({min_pq_points}). "
            f"PQ codebook training requires at least {min_pq_points} points. Use nbits <= {int(np.floor(np.log2(num_training_vectors)))}."
        )

    # 5. Recommendation warnings
    recommended_min_train = max(min_pq_points, 30 * nlist)
    if num_training_vectors < recommended_min_train:
        logger.warning(
            "Training vector count (%d) is below recommended ratio (30 * nlist = %d). "
            "Index centroids may be suboptimal.",
            num_training_vectors,
            recommended_min_train,
        )


class IVFPQIndex(BaseVectorIndex):
    """FAISS IVF-PQ Index using Inner Product similarity and IndexIDMap2."""

    def __init__(
        self,
        d: int = 512,
        nlist: int = 1024,
        m: int = 64,
        nbits: int = 8,
        nprobe: int = 16,
    ):
        self.d = d
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.nprobe = nprobe
        self.metric = faiss.METRIC_INNER_PRODUCT

        # Internal FAISS objects
        self._quantizer: Optional[faiss.IndexFlatIP] = None
        self._ivfpq: Optional[faiss.IndexIVFPQ] = None
        self._index: Optional[faiss.IndexIDMap2] = None

        self._trained = False
        self._train_stats: Dict[str, Any] = {}

    def _init_index_structures(self) -> None:
        """Instantiate empty FAISS objects."""
        self._quantizer = faiss.IndexFlatIP(self.d)
        self._ivfpq = faiss.IndexIVFPQ(
            self._quantizer,
            self.d,
            self.nlist,
            self.m,
            self.nbits,
            self.metric,
        )
        self._index = faiss.IndexIDMap2(self._ivfpq)
        self.set_nprobe(self.nprobe)

    def set_nprobe(self, nprobe: int) -> None:
        """Set query-time nprobe parameter on the underlying IVF index."""
        self.nprobe = nprobe
        if self._index is not None:
            # Extract underlying IndexIVFPQ from IndexIDMap2
            sub_index = faiss.downcast_index(self._index.index)
            sub_index.nprobe = nprobe

    @property
    def is_trained(self) -> bool:
        return self._trained and (self._index is not None and self._index.is_trained)

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index is not None else 0

    def train(self, vectors: np.ndarray) -> None:
        """Train coarse quantizer and PQ subquantizers on representative normalized vectors."""
        num_vecs, dim = vectors.shape
        if dim != self.d:
            raise ValueError(f"Vector dimension ({dim}) does not match index dimension ({self.d})")

        validate_ivfpq_parameters(
            d=self.d,
            nlist=self.nlist,
            m=self.m,
            nbits=self.nbits,
            num_training_vectors=num_vecs,
        )

        logger.info(
            "Training IVF-PQ index (d=%d, nlist=%d, m=%d, nbits=%d) with %d vectors...",
            self.d,
            self.nlist,
            self.m,
            self.nbits,
            num_vecs,
        )

        start_time = time.time()
        self._init_index_structures()

        # FAISS expects float32 C-contiguous arrays
        vectors_f32 = np.ascontiguousarray(vectors, dtype=np.float32)
        with faiss_single_thread_on_macos():
            self._index.train(vectors_f32)

        elapsed = time.time() - start_time
        self._trained = True

        self._train_stats = {
            "d": self.d,
            "nlist": self.nlist,
            "m": self.m,
            "nbits": self.nbits,
            "metric": "METRIC_INNER_PRODUCT",
            "training_sample_size": num_vecs,
            "train_time_seconds": round(elapsed, 3),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        logger.info("IVF-PQ training completed in %.2f seconds.", elapsed)

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Add vectors with explicit 64-bit integer IDs."""
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors.")
        if len(vectors) != len(ids):
            raise ValueError(f"Vectors count ({len(vectors)}) != IDs count ({len(ids)})")

        vectors_f32 = np.ascontiguousarray(vectors, dtype=np.float32)
        ids_i64 = np.ascontiguousarray(ids, dtype=np.int64)

        with faiss_single_thread_on_macos():
            self._index.add_with_ids(vectors_f32, ids_i64)
        logger.debug("Added %d vectors to index. New total: %d", len(vectors), self._index.ntotal)

    def search(self, query_vectors: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Query index for Top-K nearest neighbours using inner product similarity."""
        if self._index is None or self._index.ntotal == 0:
            return np.empty((len(query_vectors), 0), dtype=np.float32), np.empty(
                (len(query_vectors), 0), dtype=np.int64
            )

        self.set_nprobe(self.nprobe)
        queries_f32 = np.ascontiguousarray(query_vectors, dtype=np.float32)

        effective_k = min(top_k, self._index.ntotal)
        with faiss_single_thread_on_macos():
            scores, ids = self._index.search(queries_f32, effective_k)
        return scores, ids

    def save(self, index_path: Path) -> None:
        """Save FAISS index and configuration metadata atomically."""
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        temp_index_path = index_path.with_suffix(f"{index_path.suffix}.tmp.{os.getpid()}")
        try:
            faiss.write_index(self._index, str(temp_index_path))
            os.replace(temp_index_path, index_path)
        finally:
            if temp_index_path.exists():
                try:
                    temp_index_path.unlink()
                except OSError:
                    pass

        # Save metadata companion
        meta_path = index_path.with_suffix(".meta.json")
        meta = {
            "d": self.d,
            "nlist": self.nlist,
            "m": self.m,
            "nbits": self.nbits,
            "nprobe": self.nprobe,
            "metric": "METRIC_INNER_PRODUCT",
            "total_vectors": self.total_vectors,
            "is_trained": self.is_trained,
            "train_stats": self._train_stats,
            "index_size_bytes": index_path.stat().st_size if index_path.is_file() else None,
        }
        atomic_save_json(meta, meta_path)
        logger.info("Saved IVF-PQ index (%d vectors) to %s", self.total_vectors, index_path)

    def load(self, index_path: Path) -> None:
        """Load FAISS index and configuration metadata."""
        index_path = Path(index_path)
        if not index_path.is_file():
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")

        logger.info("Loading FAISS index from %s", index_path)
        self._index = faiss.read_index(str(index_path))
        self._trained = self._index.is_trained

        # Read companion metadata if available
        meta_path = index_path.with_suffix(".meta.json")
        if meta_path.is_file():
            meta = load_json(meta_path)
            self.d = meta.get("d", self.d)
            self.nlist = meta.get("nlist", self.nlist)
            self.m = meta.get("m", self.m)
            self.nbits = meta.get("nbits", self.nbits)
            self.nprobe = meta.get("nprobe", self.nprobe)
            self._train_stats = meta.get("train_stats", {})

        self.set_nprobe(self.nprobe)
        logger.info("Loaded IVF-PQ index with %d vectors (nprobe=%d).", self.total_vectors, self.nprobe)

    def reconstruct(self, key: int) -> np.ndarray:
        """Reconstruct an approximation of the vector from its int64 ID."""
        if self._index is None:
            raise RuntimeError("Index not loaded.")
        try:
            return self._index.reconstruct(int(key))
        except (RuntimeError, Exception):
            if hasattr(self._index, "index"):
                down = faiss.downcast_index(self._index.index)
                if hasattr(down, "make_direct_map"):
                    down.make_direct_map()
            return self._index.reconstruct(int(key))

