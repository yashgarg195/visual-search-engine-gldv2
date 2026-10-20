"""Unit Tests for Multi-Index Fusion (RRF) & Hybrid Ensemble Search."""

import numpy as np
import pytest

from src.index.fusion import reciprocal_rank_fusion


def test_rrf_overlapping_lists():
    """Verify that items appearing in multiple indices get boosted by consensus."""
    # Index 1 (e.g. IVF): IDs [10, 20, 30]
    res1 = ("ivfpq", np.array([0.85, 0.75, 0.65], dtype=np.float32), np.array([10, 20, 30], dtype=np.int64))
    # Index 2 (e.g. HNSW): IDs [20, 40, 10]
    res2 = ("hnswpq", np.array([0.80, 0.70, 0.60], dtype=np.float32), np.array([20, 40, 10], dtype=np.int64))

    fused_scores, fused_ids, meta = reciprocal_rank_fusion([res1, res2], k=60)

    # Candidate 20 (rank 2 in IVF, rank 1 in HNSW) should receive highest RRF consensus
    assert fused_ids[0] == 20
    assert meta[20]["consensus"] is True
    assert set(meta[20]["sources"]) == {"ivfpq", "hnswpq"}

    # Candidate 10 (rank 1 in IVF, rank 3 in HNSW) should be second
    assert fused_ids[1] == 10
    assert meta[10]["consensus"] is True

    # Candidates 40 and 30 are single-source
    assert meta[40]["consensus"] is False
    assert meta[30]["consensus"] is False
    assert set(fused_ids) == {10, 20, 30, 40}


def test_rrf_disjoint_lists():
    """Verify RRF behavior when indices return completely disjoint sets."""
    res1 = ("ivfpq", np.array([0.9, 0.8], dtype=np.float32), np.array([101, 102], dtype=np.int64))
    res2 = ("hnswpq", np.array([0.95, 0.85], dtype=np.float32), np.array([201, 202], dtype=np.int64))

    fused_scores, fused_ids, meta = reciprocal_rank_fusion([res1, res2], k=60)

    assert len(fused_ids) == 4
    for fid in fused_ids:
        assert meta[int(fid)]["consensus"] is False


def test_rrf_truncation():
    """Verify top_k parameter truncates the fused results properly."""
    res1 = ("ivf", np.array([0.9, 0.8, 0.7], dtype=np.float32), np.array([1, 2, 3], dtype=np.int64))
    res2 = ("hnsw", np.array([0.9, 0.8, 0.7], dtype=np.float32), np.array([4, 5, 6], dtype=np.int64))

    _, fused_ids, _ = reciprocal_rank_fusion([res1, res2], top_k=3)
    assert len(fused_ids) == 3


def test_rrf_empty_results():
    """Verify RRF gracefully handles empty inputs."""
    res1 = ("ivf", np.array([], dtype=np.float32), np.array([], dtype=np.int64))
    res2 = ("hnsw", np.array([], dtype=np.float32), np.array([], dtype=np.int64))

    scores, ids, meta = reciprocal_rank_fusion([res1, res2])
    assert len(ids) == 0
    assert len(scores) == 0
    assert meta == {}
