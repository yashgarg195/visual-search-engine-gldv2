"""Two-Stage Retrieval: IVF-PQ Fast Candidate Filtering + Memory-Mapped Exact Reranking."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("reranker")


class TwoStageReranker:
    """Reranks candidate results using full-precision vectors loaded on demand via memory mapping."""

    def __init__(self, embedding_dir: Path, id_mapping_df: pd.DataFrame):
        self.embedding_dir = Path(embedding_dir)
        self.id_mapping_df = id_mapping_df.set_index("faiss_id")
        self._mmap_cache: Dict[str, np.ndarray] = {}

    def _get_shard_mmap(self, shard_name: str) -> Optional[np.ndarray]:
        """Retrieve or cache memory-mapped array for a given shard."""
        if shard_name not in self._mmap_cache:
            emb_file = self.embedding_dir / f"{shard_name}_emb.npy"
            if not emb_file.is_file():
                logger.warning("Shard vector file not found for reranking: %s", emb_file)
                return None
            self._mmap_cache[shard_name] = np.load(emb_file, mmap_mode="r")
        return self._mmap_cache[shard_name]

    def rerank_candidates(
        self,
        query_vector: np.ndarray,
        candidate_faiss_ids: List[int],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Compute exact dot product for candidates and return the top-K reranked list."""
        if len(candidate_faiss_ids) == 0:
            return []

        # Ensure query is 1D normalized float32
        query_vec = np.squeeze(np.asarray(query_vector, dtype=np.float32))
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        reranked_items = []
        for fid in candidate_faiss_ids:
            if fid not in self.id_mapping_df.index:
                continue

            row = self.id_mapping_df.loc[fid]
            shard_name = str(row.get("shard_name", ""))
            row_idx = int(row.get("shard_row_idx", -1))

            if not shard_name or row_idx < 0:
                continue

            arr = self._get_shard_mmap(shard_name)
            if arr is None or row_idx >= len(arr):
                continue

            # Load only this single vector from memory map
            full_precision_vec = np.array(arr[row_idx], copy=True, dtype=np.float32)
            exact_score = float(np.dot(query_vec, full_precision_vec))

            reranked_items.append({
                "faiss_id": fid,
                "exact_score": exact_score,
                "image_id": str(row.get("image_id", "")),
                "landmark_id": str(row.get("landmark_id", "")),
                "split": str(row.get("split", "")),
            })

        # Sort descending by exact inner product
        reranked_items.sort(key=lambda x: x["exact_score"], reverse=True)
        return reranked_items[:top_k]
