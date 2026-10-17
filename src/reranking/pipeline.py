"""Pluggable Stage-2 Geometric Reranking Pipeline.

Reranks coarse candidate matches from FAISS IVF-PQ using local keypoint
geometric verification. Fully decoupled and non-destructive: if disabled or
in case of failure, candidates preserve their original coarse retrieval order.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from src.reranking.geometric_verifier import GeometricVerifier, GeometricVerificationResult
from src.utils.logging import get_logger

logger = get_logger("geometric_reranker")


class GeometricReranker:
    """Orchestrates candidate image loading, local keypoint extraction, and geometric re-scoring."""

    def __init__(
        self,
        verifier: Optional[GeometricVerifier] = None,
        verification_boost: float = 1.0,
        alpha: Optional[float] = None,
        max_cache_size: int = 500,
    ):
        """Args:
            verifier: Instance of GeometricVerifier (defaults to SIFT).
            verification_boost: Multiplier for geometric verification boost.
            alpha: Legacy/optional blend weight (if passed, adjusts boost).
            max_cache_size: Number of candidate image descriptors to hold in LRU memory cache.
        """
        self.verifier = verifier or GeometricVerifier()
        if alpha is not None:
            # Map legacy alpha to verification_boost if specified
            self.verification_boost = float((1.0 - alpha) / max(alpha, 0.01))
        else:
            self.verification_boost = float(max(0.0, verification_boost))
        self.max_cache_size = max_cache_size
        self._cache: OrderedDict[str, Tuple[np.ndarray, np.ndarray, Tuple[int, int]]] = OrderedDict()

    def _get_candidate_features(
        self, img_path: Union[str, Path]
    ) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[int, int]]]:
        """Fetch or extract candidate keypoints with an LRU cache."""
        path_str = str(img_path)
        if path_str in self._cache:
            self._cache.move_to_end(path_str)
            return self._cache[path_str]

        p = Path(path_str)
        if not p.is_file():
            return None

        try:
            feats = self.verifier.extract_features(p)
            if len(self._cache) >= self.max_cache_size:
                self._cache.popitem(last=False)
            self._cache[path_str] = feats
            return feats
        except Exception as e:
            logger.debug("Failed extracting features for candidate %s: %s", path_str, e)
            return None

    def rerank(
        self,
        query_image: Optional[Union[str, Path, Image.Image, np.ndarray]],
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        rerank_depth: int = 50,
    ) -> List[Dict[str, Any]]:
        """Perform geometric verification reranking on coarse search results.

        If query_image is None or empty, returns candidates unchanged.
        """
        if query_image is None or not candidates:
            return candidates

        # 1. Extract query features
        try:
            q_kpts, q_descs, q_shape = self.verifier.extract_features(query_image)
        except Exception as e:
            logger.warning("Failed extracting query features for geometric reranking: %s. Returning coarse results.", e)
            return candidates

        if len(q_kpts) < 4:
            logger.info("Query has too few keypoints (%d). Skipping geometric verification.", len(q_kpts))
            return candidates

        # 2. Slice candidates to rerank_depth
        pool_to_rerank = candidates[:rerank_depth]
        untouched = candidates[rerank_depth:]

        reranked_pool: List[Dict[str, Any]] = []

        for candidate in pool_to_rerank:
            cand_copy = dict(candidate)
            img_path = cand_copy.get("image_path")
            coarse_score = float(cand_copy.get("score", 0.0))

            cand_feats = self._get_candidate_features(img_path) if img_path else None

            if cand_feats is None:
                # Fall back to original coarse score
                cand_copy["geometric_verified"] = False
                cand_copy["inliers"] = 0
                cand_copy["matches"] = 0
                cand_copy["inlier_ratio"] = 0.0
                cand_copy["geometric_score"] = 0.0
                cand_copy["final_score"] = coarse_score
            else:
                c_kpts, c_descs, c_shape = cand_feats
                res: GeometricVerificationResult = self.verifier.verify_pair(
                    q_kpts, q_descs, q_shape, c_kpts, c_descs, c_shape
                )

                cand_copy["geometric_verified"] = res.is_verified
                cand_copy["inliers"] = res.num_inliers
                cand_copy["matches"] = res.num_matches
                cand_copy["inlier_ratio"] = res.inlier_ratio
                cand_copy["spatial_dispersion"] = res.spatial_dispersion
                cand_copy["geometric_score"] = res.geometric_score

                if res.is_verified:
                    # Positive geometric verification boosts coarse score proportionally
                    final = coarse_score * (1.0 + self.verification_boost * res.geometric_score)
                else:
                    # Penalize unverified candidates slightly
                    final = coarse_score * 0.95

                cand_copy["final_score"] = round(float(final), 4)

            reranked_pool.append(cand_copy)

        # 3. Sort pool by final_score descending
        reranked_pool.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

        # 4. Re-assign ranks
        full_results = reranked_pool + untouched
        for r_idx, match in enumerate(full_results, start=1):
            match["rank"] = r_idx

        if top_k is not None:
            full_results = full_results[:top_k]

        return full_results
