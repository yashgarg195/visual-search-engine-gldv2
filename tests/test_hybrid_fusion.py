"""Unit tests for Hybrid Multi-Signal Fusion Reranker."""

import pytest
from src.reranking.hybrid_fusion import HybridMultiSignalReranker, haversine_distance_km


def test_haversine_distance():
    # Distance between Eiffel Tower (48.8584, 2.2945) and Arc de Triomphe (48.8738, 2.2950) ~ 1.7 km
    dist = haversine_distance_km(48.8584, 2.2945, 48.8738, 2.2950)
    assert 1.5 < dist < 2.0

    # Distance to same point is 0
    assert haversine_distance_km(48.8584, 2.2945, 48.8584, 2.2945) == 0.0


def test_hybrid_fusion_reranker_basic():
    reranker = HybridMultiSignalReranker(weight_visual=0.50, weight_vicinity=0.30, weight_spatial=0.20, spatial_sigma_km=5.0)

    # Synthetic candidates:
    # Match 1: High visual similarity, far away in Moscow
    # Match 2 & 3: Moderate visual similarity, in Paris around Eiffel Tower
    matches = [
        {
            "rank": 1,
            "landmark_id": "1",
            "landmark_name": "Saint Basil's",
            "score": 0.85,
            "latitude": 55.7525,
            "longitude": 37.6231,
            "hierarchical_label": "Russia > Moscow > Red Square",
        },
        {
            "rank": 2,
            "landmark_id": "2",
            "landmark_name": "Eiffel Tower",
            "score": 0.80,
            "latitude": 48.8584,
            "longitude": 2.2945,
            "hierarchical_label": "France > Paris > Eiffel Tower",
        },
        {
            "rank": 3,
            "landmark_id": "2",
            "landmark_name": "Eiffel Tower Champ de Mars",
            "score": 0.78,
            "latitude": 48.8560,
            "longitude": 2.2980,
            "hierarchical_label": "France > Paris > Eiffel Tower",
        },
    ]

    # Rerank with Paris Eiffel Tower anchor
    reranked = reranker.rerank(
        retrieved_matches=matches,
        prominent_anchor="Eiffel Tower",
        anchor_coords=(48.8584, 2.2945),
    )

    assert len(reranked) == 3
    # Eiffel Tower cluster should beat Saint Basil's due to vicinity consensus + spatial proximity
    top_match = reranked[0]
    assert "Eiffel Tower" in top_match["landmark_name"]
    assert top_match["is_vicinity_verified"] is True
    assert top_match["final_score"] > reranked[-1]["final_score"]
    assert "final_score" in top_match
    assert "visual_sim" in top_match
    assert "vicinity_score" in top_match
    assert "spatial_score" in top_match


def test_hybrid_fusion_reranker_graceful_fallback():
    reranker = HybridMultiSignalReranker()

    # Matches with missing coordinates
    matches = [
        {
            "rank": 1,
            "landmark_id": "10",
            "landmark_name": "Grand Canyon",
            "score": 0.90,
            "latitude": None,
            "longitude": None,
        },
        {
            "rank": 2,
            "landmark_id": "20",
            "landmark_name": "Yellowstone",
            "score": 0.85,
            "latitude": 0.0,
            "longitude": 0.0,
        },
    ]

    reranked = reranker.rerank(retrieved_matches=matches)
    assert len(reranked) == 2
    assert reranked[0]["rank"] == 1
    assert reranked[1]["rank"] == 2
    assert reranked[0]["final_score"] >= reranked[1]["final_score"]
