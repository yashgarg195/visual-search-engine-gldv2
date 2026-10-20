"""Multi-Index Fusion & Reciprocal Rank Fusion (RRF).

Combines candidate retrieval results from distinct index architectures
(e.g., IVF-PQ Voronoi clustering and HNSW-PQ graph exploration) into a single,
high-precision consensus ranking.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def reciprocal_rank_fusion(
    results: List[Tuple[str, np.ndarray, np.ndarray]],
    k: int = 60,
    top_k: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, Dict[str, Any]]]:
    """Combine multiple ranked search result lists using Reciprocal Rank Fusion (RRF).

    Args:
        results: List of (source_name, scores_array, ids_array) tuples for 1D arrays of scores/ids.
        k: Smoothing constant for RRF (default 60, Cormack et al.).
        top_k: Optional truncation length for final merged list.

    Returns:
        fused_scores: (M,) float32 array of merged raw similarity scores.
        fused_ids: (M,) int64 array of unique candidate IDs ordered by RRF consensus.
        fusion_meta: Dict mapping fid -> {"rrf_score": float, "sources": list, "consensus": bool}
    """
    candidates: Dict[int, Dict[str, Any]] = {}

    for source_name, scores, ids in results:
        for rank_idx, (score, fid) in enumerate(zip(scores, ids), start=1):
            fid_int = int(fid)
            if fid_int < 0:
                continue

            rrf_delta = 1.0 / (k + rank_idx)

            if fid_int not in candidates:
                candidates[fid_int] = {
                    "fid": fid_int,
                    "rrf_score": 0.0,
                    "max_score": float(score),
                    "score_sum": float(score),
                    "appearances": 0,
                    "sources": [],
                    "ranks": {},
                }

            entry = candidates[fid_int]
            entry["rrf_score"] += rrf_delta
            entry["appearances"] += 1
            entry["sources"].append(source_name)
            entry["ranks"][source_name] = rank_idx
            if float(score) > entry["max_score"]:
                entry["max_score"] = float(score)
            if entry["appearances"] > 1:
                entry["score_sum"] += float(score)

    total_sources = len(results)

    # Sort candidates primarily by RRF score descending, breaking ties by max raw cosine score
    sorted_candidates = sorted(
        candidates.values(),
        key=lambda x: (x["rrf_score"], x["max_score"]),
        reverse=True,
    )

    if top_k is not None:
        sorted_candidates = sorted_candidates[:top_k]

    fused_ids_list = []
    fused_scores_list = []
    fusion_meta: Dict[int, Dict[str, Any]] = {}

    for c in sorted_candidates:
        fid = c["fid"]
        fused_ids_list.append(fid)
        # Use max raw score as the displayed similarity
        fused_scores_list.append(c["max_score"])
        fusion_meta[fid] = {
            "rrf_score": round(c["rrf_score"], 6),
            "sources": c["sources"],
            "consensus": (len(c["sources"]) >= 2 and total_sources >= 2),
            "ranks": c["ranks"],
        }

    fused_ids = np.array(fused_ids_list, dtype=np.int64)
    fused_scores = np.array(fused_scores_list, dtype=np.float32)

    return fused_scores, fused_ids, fusion_meta
