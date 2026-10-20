"""Hybrid Multi-Signal Fusion Reranker (Production Architecture).

Combines:
1. Visual Similarity (OpenCLIP / FAISS cosine similarity) - 50%
2. Vicinity & Neighborhood Consensus (Area clustering & taxonomy agreement) - 30%
3. Geospatial Proximity (Haversine distance to cluster centroid) - 20%

Formula:
    FinalScore = 0.50 * VisualSim + 0.30 * VicinityConsensus + 0.20 * SpatialProximity

Graceful Fallbacks:
- If GPS coordinates are missing or (0, 0), falls back to Taxonomy & Vicinity Consensus.
- If LLM rate limits hit, operates entirely on local taxonomy tree and metadata coordinates.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging import get_logger

logger = get_logger("hybrid_fusion")


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the great-circle distance between two GPS points in kilometers."""
    r_earth = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return r_earth * c


class HybridMultiSignalReranker:
    """Reranks visual search results by fusing visual similarity, physical vicinity consensus, and GPS proximity."""

    def __init__(
        self,
        weight_visual: float = 0.50,
        weight_vicinity: float = 0.30,
        weight_spatial: float = 0.20,
        spatial_sigma_km: float = 5.0,
    ):
        self.w_visual = weight_visual
        self.w_vicinity = weight_vicinity
        self.w_spatial = weight_spatial
        self.spatial_sigma_km = spatial_sigma_km

    def _extract_vicinity_key(self, match: Dict[str, Any], prominent_anchor: Optional[str] = None) -> str:
        """Extract a canonical physical area / neighborhood key for a candidate match."""
        name = str(match.get("landmark_name") or "").lower().strip()
        hier = str(match.get("hierarchical_label") or "").strip()

        # If hierarchical taxonomy exists (e.g., 'Europe > Russia > Moscow > Red Square')
        if hier and hier.lower() != "nan":
            parts = [p.strip() for p in hier.split(">") if p.strip()]
            if len(parts) >= 2:
                # Use bottom 2 levels (e.g. City + Landmark/Area)
                return f"{parts[-2]} > {parts[-1]}".lower()
            elif len(parts) == 1:
                return parts[0].lower()

        # Check prominent anchor match
        if prominent_anchor:
            clean_anchor = prominent_anchor.lower().strip()
            if clean_anchor in name or name in clean_anchor:
                return clean_anchor

        # Clean landmark name as fallback cluster key
        clean_name = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", name)).strip()
        return clean_name or "unknown_vicinity"

    def rerank(
        self,
        retrieved_matches: List[Dict[str, Any]],
        prominent_anchor: Optional[str] = None,
        anchor_coords: Optional[Tuple[float, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute hybrid multi-signal fusion reranking across retrieved candidate matches."""
        if not retrieved_matches:
            return []

        # 1. Normalize Visual Scores into [0.0, 1.0]
        raw_scores = [float(m.get("score", 0.0)) for m in retrieved_matches]
        min_s = min(raw_scores)
        max_s = max(raw_scores)
        score_range = max(1e-5, max_s - min_s)

        visual_sims: List[float] = []
        for s in raw_scores:
            # Calibrate cosine similarity (~0.25 to ~0.75 range)
            norm_s = min(1.0, max(0.0, (s - 0.20) / 0.60))
            # Relative rank preservation
            rel_s = (s - min_s) / score_range
            visual_sims.append(0.6 * norm_s + 0.4 * rel_s)

        # 2. Group into Physical Vicinity / Neighborhood Clusters
        cluster_weights: Dict[str, float] = {}
        cluster_matches: Dict[str, List[int]] = {}
        cluster_coords: Dict[str, List[Tuple[float, float]]] = {}

        for idx, match in enumerate(retrieved_matches):
            v_key = self._extract_vicinity_key(match, prominent_anchor=prominent_anchor)
            match["_vicinity_key"] = v_key
            v_weight = visual_sims[idx]

            cluster_weights[v_key] = cluster_weights.get(v_key, 0.0) + v_weight
            if v_key not in cluster_matches:
                cluster_matches[v_key] = []
                cluster_coords[v_key] = []
            cluster_matches[v_key].append(idx)

            # Collect valid GPS coordinates
            try:
                lat = float(match.get("latitude") or 0.0)
                lon = float(match.get("longitude") or 0.0)
                if abs(lat) > 1e-4 and abs(lon) > 1e-4:
                    cluster_coords[v_key].append((lat, lon))
            except (ValueError, TypeError):
                pass

        # Identify winning consensus cluster
        total_cluster_weight = max(1e-5, sum(cluster_weights.values()))
        winning_cluster = max(cluster_weights.keys(), key=lambda k: cluster_weights[k])
        max_cluster_weight = cluster_weights[winning_cluster]

        # Determine reference GPS centroid for winning cluster
        centroid_lat: Optional[float] = None
        centroid_lon: Optional[float] = None

        if anchor_coords and abs(anchor_coords[0]) > 1e-4:
            centroid_lat, centroid_lon = anchor_coords
        elif cluster_coords.get(winning_cluster):
            valid_pts = cluster_coords[winning_cluster]
            centroid_lat = sum(p[0] for p in valid_pts) / float(len(valid_pts))
            centroid_lon = sum(p[1] for p in valid_pts) / float(len(valid_pts))

        # 3. Compute Multi-Signal Scores for Each Candidate
        reranked_results: List[Dict[str, Any]] = []

        for idx, match in enumerate(retrieved_matches):
            v_sim = visual_sims[idx]
            v_key = match.get("_vicinity_key", "")

            # Signal 2: Vicinity Consensus Score [0.0, 1.0]
            c_weight = cluster_weights.get(v_key, 0.0)
            is_winning_vicinity = (v_key == winning_cluster)
            vicinity_score = min(1.0, c_weight / total_cluster_weight * 1.5) if is_winning_vicinity else (c_weight / total_cluster_weight * 0.5)

            # Signal 3: Spatial Proximity Score [0.0, 1.0]
            dist_km: Optional[float] = None
            spatial_score = vicinity_score  # Default fallback if coordinates unavailable

            if centroid_lat is not None and centroid_lon is not None:
                try:
                    c_lat = float(match.get("latitude") or 0.0)
                    c_lon = float(match.get("longitude") or 0.0)
                    if abs(c_lat) > 1e-4 and abs(c_lon) > 1e-4:
                        dist_km = haversine_distance_km(c_lat, c_lon, centroid_lat, centroid_lon)
                        spatial_score = float(math.exp(-(dist_km ** 2) / (2.0 * (self.spatial_sigma_km ** 2))))
                    else:
                        # GPS missing on this record -> graceful fallback to vicinity score
                        spatial_score = vicinity_score
                except (ValueError, TypeError):
                    spatial_score = vicinity_score

            # Combined Fusion Formula
            final_score = (
                self.w_visual * v_sim
                + self.w_vicinity * vicinity_score
                + self.w_spatial * spatial_score
            )

            # Build enriched match record
            m_copy = dict(match)
            m_copy.pop("_vicinity_key", None)
            m_copy["final_score"] = round(float(final_score), 4)
            m_copy["visual_sim"] = round(float(v_sim), 4)
            m_copy["vicinity_score"] = round(float(vicinity_score), 4)
            m_copy["spatial_score"] = round(float(spatial_score), 4)
            m_copy["is_vicinity_verified"] = bool(dist_km is not None and dist_km <= 2.0)
            m_copy["distance_km"] = round(float(dist_km), 2) if dist_km is not None else None
            m_copy["vicinity_cluster"] = v_key.title() if v_key else "Landmark Area"

            reranked_results.append(m_copy)

        # 4. Sort descending by fused FinalScore
        reranked_results.sort(key=lambda m: m["final_score"], reverse=True)

        # Re-assign sequential ranks
        for rank_idx, m in enumerate(reranked_results, start=1):
            m["rank"] = rank_idx

        logger.info(
            "Hybrid Fusion Rerank complete across %d candidates. Winning Vicinity: '%s' (Consensus: %.1f%%)",
            len(reranked_results),
            winning_cluster.title(),
            (max_cluster_weight / total_cluster_weight) * 100.0,
        )

        return reranked_results
