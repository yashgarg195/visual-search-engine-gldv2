"""Pluggable Stage-2 Geometric Reranking Pipeline.

Reranks coarse candidate matches from FAISS IVF-PQ or HNSW-PQ using local keypoint
geometric verification (SIFT or SuperPoint + LightGlue).
Fully decoupled and non-destructive: if disabled or in case of missing images,
candidates preserve their original coarse retrieval order.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from src.reranking.geometric_verifier import GeometricVerifier, GeometricVerificationResult
from src.reranking.thumbnail_fetcher import get_thumbnail_fetcher
from src.utils.logging import get_logger

logger = get_logger("geometric_reranker")

try:
    from src.reranking.lightglue_verifier import LightGlueVerifier
    HAS_LIGHTGLUE_VERIFIER = True
except ImportError:
    HAS_LIGHTGLUE_VERIFIER = False


class GeometricReranker:
    """Orchestrates candidate image loading, local keypoint extraction, and geometric re-scoring."""

    def __init__(
        self,
        verifier: Optional[Any] = None,
        matcher_type: str = "sift",
        verification_boost: float = 1.0,
        alpha: Optional[float] = None,
        max_cache_size: int = 500,
        fetch_thumbnails: bool = True,
    ):
        """Args:
            verifier: Instance of GeometricVerifier or LightGlueVerifier.
            matcher_type: Matcher algorithm ('sift', 'orb', or 'lightglue').
            verification_boost: Multiplier for geometric verification boost.
            alpha: Legacy/optional blend weight (if passed, adjusts boost).
            max_cache_size: Number of candidate image descriptors to hold in LRU memory cache.
            fetch_thumbnails: Whether to attempt on-demand thumbnail fetching if local image is missing.
        """
        self.matcher_type = matcher_type.lower()
        self.fetch_thumbnails = fetch_thumbnails

        if verifier is not None:
            self.verifier = verifier
        elif self.matcher_type in ("lightglue", "superpoint", "neural"):
            if HAS_LIGHTGLUE_VERIFIER:
                try:
                    self.verifier = LightGlueVerifier()
                except Exception as e:
                    logger.warning("Failed initializing LightGlueVerifier: %s. Falling back to SIFT.", e)
                    self.verifier = GeometricVerifier(detector_type="sift")
            else:
                logger.warning("LightGlueVerifier module not found. Falling back to SIFT.")
                self.verifier = GeometricVerifier(detector_type="sift")
        else:
            self.verifier = GeometricVerifier(detector_type=self.matcher_type)

        if alpha is not None:
            self.verification_boost = float((1.0 - alpha) / max(alpha, 0.01))
        else:
            self.verification_boost = float(max(0.0, verification_boost))

        self.max_cache_size = max_cache_size
        self._cache: OrderedDict[str, Tuple[np.ndarray, Any, Tuple[int, int]]] = OrderedDict()

    def _get_candidate_features(
        self, candidate: Dict[str, Any]
    ) -> Optional[Tuple[np.ndarray, Any, Tuple[int, int]]]:
        """Fetch or extract candidate keypoints with an LRU cache."""
        img_path = candidate.get("image_path")
        cache_key = str(img_path) if img_path else ""

        if cache_key and cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        # 1. Try local file path first
        if img_path:
            p = Path(img_path)
            if p.is_file():
                try:
                    feats = self.verifier.extract_features(p)
                    if len(self._cache) >= self.max_cache_size:
                        self._cache.popitem(last=False)
                    self._cache[cache_key] = feats
                    return feats
                except Exception as e:
                    logger.debug("Failed extracting features for %s: %s", img_path, e)

        # 2. On-demand thumbnail fetch if enabled and image URL exists
        cand_url = candidate.get("image_url") or candidate.get("thumbnail_url")
        if self.fetch_thumbnails and cand_url and isinstance(cand_url, str) and cand_url.startswith("http"):
            url_key = f"url:{cand_url}"
            if url_key in self._cache:
                self._cache.move_to_end(url_key)
                return self._cache[url_key]

            fetcher = get_thumbnail_fetcher()
            fetched_img = fetcher.fetch(cand_url)
            if fetched_img is not None:
                try:
                    feats = self.verifier.extract_features(fetched_img)
                    if len(self._cache) >= self.max_cache_size:
                        self._cache.popitem(last=False)
                    self._cache[url_key] = feats
                    return feats
                except Exception as e:
                    logger.debug("Failed extracting features from fetched image %s: %s", cand_url, e)

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
            coarse_score = float(cand_copy.get("score", 0.0))

            cand_feats = self._get_candidate_features(cand_copy)

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
                    final = coarse_score * (1.0 + self.verification_boost * res.geometric_score)
                else:
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
