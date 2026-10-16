"""Similarity-Weighted Voting for Top-K Landmark Candidate Identification.

Aggregates individual retrieval match scores across candidate landmarks.
Scores are reported strictly as inner product / cosine similarities,
never as calibrated probabilities.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LandmarkVoteSummary:
    """Consolidated voting result for a single candidate landmark."""

    split: str
    landmark_id: str
    landmark_name: str
    is_resolved: bool
    total_vote_score: float
    match_count: int
    highest_score: float
    category_url: Optional[str] = None
    hierarchical_label: Optional[str] = None
    supporting_images: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class VotingResult:
    """Overall outcome of similarity-weighted voting across Top-K retrieval matches."""

    leading_candidate: Optional[LandmarkVoteSummary]
    second_candidate: Optional[LandmarkVoteSummary]
    margin: float
    all_candidates: List[LandmarkVoteSummary]
    total_retrieved: int
    note: str = (
        "Similarity scores represent cosine similarity in [-1, 1] for normalized embeddings, "
        "not calibrated statistical probabilities."
    )


def compute_similarity_weighted_votes(
    retrieved_matches: List[Dict[str, Any]],
) -> VotingResult:
    """Compute similarity-weighted voting across a list of retrieved Top-K match dictionaries.

    Each match is expected to contain:
        - 'score': float (cosine similarity)
        - 'split': str
        - 'landmark_id': str
        - 'landmark_name': str
        - 'image_id': str
        - (optional) 'category_url', 'hierarchical_label', 'is_resolved'
    """
    if not retrieved_matches:
        return VotingResult(
            leading_candidate=None,
            second_candidate=None,
            margin=0.0,
            all_candidates=[],
            total_retrieved=0,
        )

    # Group by (split, landmark_id)
    candidates_map: Dict[tuple, Dict[str, Any]] = {}

    for match in retrieved_matches:
        split = str(match.get("split", "train")).strip().lower()
        lid = str(match.get("landmark_id", "unknown")).strip()
        score = float(match.get("score", 0.0))
        # Use positive similarity weight
        weight = max(0.0, score)

        key = (split, lid)
        if key not in candidates_map:
            candidates_map[key] = {
                "split": split,
                "landmark_id": lid,
                "landmark_name": match.get("landmark_name", f"Landmark #{lid}"),
                "is_resolved": bool(match.get("is_resolved", False)),
                "category_url": match.get("category_url"),
                "hierarchical_label": match.get("hierarchical_label"),
                "total_vote_score": 0.0,
                "match_count": 0,
                "highest_score": -1.0,
                "supporting_images": [],
            }

        cand = candidates_map[key]
        cand["total_vote_score"] += weight
        cand["match_count"] += 1
        cand["highest_score"] = max(cand["highest_score"], score)
        cand["supporting_images"].append({
            "image_id": match.get("image_id"),
            "score": score,
            "rank": match.get("rank"),
            "image_path": match.get("image_path"),
        })

    # Sort candidates by total vote score descending
    sorted_cands = sorted(
        candidates_map.values(),
        key=lambda c: (c["total_vote_score"], c["match_count"], c["highest_score"]),
        reverse=True,
    )

    summaries = [
        LandmarkVoteSummary(
            split=c["split"],
            landmark_id=c["landmark_id"],
            landmark_name=c["landmark_name"],
            is_resolved=c["is_resolved"],
            total_vote_score=round(c["total_vote_score"], 4),
            match_count=c["match_count"],
            highest_score=round(c["highest_score"], 4),
            category_url=c["category_url"],
            hierarchical_label=c["hierarchical_label"],
            supporting_images=c["supporting_images"],
        )
        for c in sorted_cands
    ]

    leading = summaries[0] if summaries else None
    second = summaries[1] if len(summaries) > 1 else None
    margin = round(leading.total_vote_score - (second.total_vote_score if second else 0.0), 4) if leading else 0.0

    return VotingResult(
        leading_candidate=leading,
        second_candidate=second,
        margin=margin,
        all_candidates=summaries,
        total_retrieved=len(retrieved_matches),
    )
