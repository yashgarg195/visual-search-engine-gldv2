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
        # Duplicate faiss_id values are reachable if a shard is appended twice
        # by an interrupted build. With duplicates present, .loc[fid] returns a
        # DataFrame instead of a Series and every lookup below raises. Drop
        # them up front, keeping the first occurrence.
        if "faiss_id" in id_mapping_df.columns:
            dupes = int(id_mapping_df["faiss_id"].duplicated().sum())
            if dupes:
                logger.warning(
                    "ID mapping table contains %d duplicate faiss_id rows; keeping first occurrence of each.",
                    dupes,
                )
                id_mapping_df = id_mapping_df.drop_duplicates(subset="faiss_id", keep="first")
            self.id_mapping_df = id_mapping_df.set_index("faiss_id")
        else:
            # Already indexed by faiss_id.
            if id_mapping_df.index.has_duplicates:
                logger.warning("ID mapping index contains duplicates; keeping first occurrence of each.")
                id_mapping_df = id_mapping_df[~id_mapping_df.index.duplicated(keep="first")]
            self.id_mapping_df = id_mapping_df
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
        """Compute exact dot product for candidates and return the top-K reranked list using batched shard reads."""
        if len(candidate_faiss_ids) == 0:
            return []

        # Ensure query is 1D normalized float32
        query_vec = np.squeeze(np.asarray(query_vector, dtype=np.float32))
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        # Filter valid IDs and deduplicate while preserving order
        unique_fids = list(dict.fromkeys(int(f) for f in candidate_faiss_ids if f >= 0))
        if not unique_fids:
            return []

        # 1. Batched mapping lookup via reindex
        sub_df = self.id_mapping_df.reindex(unique_fids)
        sub_df = sub_df.dropna(subset=["shard_name", "shard_row_idx"])
        if sub_df.empty:
            return []

        sub_df["shard_row_idx"] = sub_df["shard_row_idx"].astype(int)
        sub_df = sub_df[sub_df["shard_row_idx"] >= 0]
        if sub_df.empty:
            return []

        # 2. Group surviving rows by shard_name
        reranked_items = []
        for shard_name, group in sub_df.groupby("shard_name", sort=False):
            arr = self._get_shard_mmap(str(shard_name))
            if arr is None:
                continue

            row_indices = group["shard_row_idx"].values
            valid_mask = row_indices < len(arr)
            if not np.any(valid_mask):
                continue
            if not np.all(valid_mask):
                group = group.iloc[valid_mask]
                row_indices = row_indices[valid_mask]

            # 3. Batch fancy-index read + matmul against query vector
            vectors = np.asarray(arr[row_indices], dtype=np.float32)
            scores = vectors @ query_vec

            fids = group.index.values
            img_ids = group["image_id"].astype(str).values if "image_id" in group.columns else [""] * len(group)
            lm_ids = group["landmark_id"].astype(str).values if "landmark_id" in group.columns else [""] * len(group)
            splits = group["split"].astype(str).values if "split" in group.columns else [""] * len(group)

            for fid, sc, iid, lid, sp in zip(fids, scores, img_ids, lm_ids, splits):
                reranked_items.append({
                    "faiss_id": int(fid),
                    "exact_score": float(sc),
                    "image_id": str(iid),
                    "landmark_id": str(lid),
                    "split": str(sp),
                })

        # 4. Sort descending by exact inner product
        reranked_items.sort(key=lambda x: x["exact_score"], reverse=True)
        return reranked_items[:top_k]
