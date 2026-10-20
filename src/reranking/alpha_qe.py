"""Alpha-Query Expansion (α-QE) for Landmark Vector Retrieval.

Implements similarity-weighted query expansion:
    q_expanded = normalize(q + sum_{i=1}^m s_i^alpha * v_i)

This technique operates purely in vector space with zero extra disk storage,
adding ~1-2 ms latency while substantially improving top-k recall and precision.
"""

from typing import List, Optional, Tuple, Union
import numpy as np

from src.index.base import BaseVectorIndex
from src.utils.logging import get_logger

logger = get_logger("alpha_qe")


class AlphaQueryExpander:
    """Performs alpha-query expansion given an initial query embedding and top candidate vectors."""

    def __init__(
        self,
        alpha: float = 3.0,
        num_candidates: int = 5,
        min_similarity: float = 0.20,
    ):
        """
        Args:
            alpha: Exponent for weighting candidate similarities (standard 3.0 in landmark literature).
            num_candidates: Number of top candidate vectors to blend into the expanded query (m).
            min_similarity: Minimum cosine similarity score for a candidate to contribute.
        """
        self.alpha = float(alpha)
        self.num_candidates = int(num_candidates)
        self.min_similarity = float(min_similarity)

    def expand_query(
        self,
        query_emb: np.ndarray,
        candidate_ids: List[int],
        candidate_scores: List[float],
        index: BaseVectorIndex,
    ) -> np.ndarray:
        """Construct the expanded query vector.

        Args:
            query_emb: (D,) L2-normalized float32 query vector.
            candidate_ids: List of FAISS candidate IDs from initial search.
            candidate_scores: List of corresponding cosine similarity scores.
            index: Vector index providing vector reconstruction.

        Returns:
            expanded_emb: (D,) L2-normalized float32 expanded query vector.
        """
        q = np.array(query_emb, dtype=np.float32).flatten()
        q_norm = float(np.linalg.norm(q))
        if q_norm > 1e-8:
            q = q / q_norm

        expanded = q.copy()
        count = 0

        for fid, score in zip(candidate_ids[: self.num_candidates], candidate_scores[: self.num_candidates]):
            if fid < 0 or score < self.min_similarity:
                continue

            try:
                cand_vec = index.reconstruct(int(fid))
                c_norm = float(np.linalg.norm(cand_vec))
                if c_norm > 1e-8:
                    cand_vec = cand_vec / c_norm
                    weight = float(score ** self.alpha)
                    expanded += weight * cand_vec
                    count += 1
            except Exception as e:
                logger.debug("Failed reconstructing vector for FAISS ID %d: %s", fid, e)
                continue

        exp_norm = float(np.linalg.norm(expanded))
        if exp_norm > 1e-8:
            expanded = expanded / exp_norm

        logger.debug(
            "Expanded query with %d / %d candidate vectors (alpha=%.1f)",
            count,
            self.num_candidates,
            self.alpha,
        )
        return expanded
