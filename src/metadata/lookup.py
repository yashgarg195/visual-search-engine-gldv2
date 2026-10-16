"""Persistent Landmark Metadata and Geolocation Lookup Table.

Provides O(1) in-memory or Parquet-backed lookup for resolved landmark details.
Keeps geographic coordinates in a distinct table to separate identification from geolocation.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd

from src.metadata.resolver import LandmarkInfo, LandmarkResolver
from src.utils.logging import get_logger

logger = get_logger("metadata_lookup")


class LandmarkLookupTable:
    """Split-aware lookup table for landmark names and optional coordinates."""

    def __init__(self):
        # Key: (split.lower(), landmark_id.strip()) -> LandmarkInfo
        self._landmark_cache: Dict[Tuple[str, str], LandmarkInfo] = {}
        # Key: image_id.strip() -> row dict
        self._image_cache: Dict[str, Dict[str, Any]] = {}
        # Key: (split.lower(), landmark_id.strip()) -> (latitude, longitude)
        self._coordinates_table: Dict[Tuple[str, str], Tuple[float, float]] = {}

    def load_from_parquet(self, parquet_path: Path) -> int:
        """Populate lookup table from a processed metadata Parquet file."""
        parquet_path = Path(parquet_path)
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Parquet metadata file not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)
        total_rows = len(df)

        # 1. Fast vectorized landmark info population (deduplicate landmarks first)
        landmark_cols = [
            c
            for c in [
                "split",
                "landmark_id",
                "landmark_name",
                "is_resolved",
                "category_url",
                "hierarchical_label",
                "supercategory",
                "natural_or_human_made",
            ]
            if c in df.columns
        ]
        df_unique_landmarks = df[landmark_cols].drop_duplicates(subset=["split", "landmark_id"])

        for r in df_unique_landmarks.to_dict(orient="records"):
            split = str(r.get("split", "train")).strip().lower()
            lid = str(r.get("landmark_id", "")).strip()
            key = (split, lid)
            if key not in self._landmark_cache:
                self._landmark_cache[key] = LandmarkInfo(
                    split=split,
                    landmark_id=lid,
                    landmark_name=str(r.get("landmark_name", f"Landmark #{lid}")),
                    is_resolved=bool(r.get("is_resolved", False)),
                    category_url=r.get("category_url"),
                    hierarchical_label=r.get("hierarchical_label"),
                    supercategory=r.get("supercategory"),
                    natural_or_human_made=r.get("natural_or_human_made"),
                )

        # 2. Fast image cache population (for datasets under 100k records, e.g. tests or small subsets)
        if total_rows <= 100_000 and "image_id" in df.columns:
            for r in df.to_dict(orient="records"):
                iid = str(r.get("image_id", "")).strip()
                if iid:
                    self._image_cache[iid] = {
                        "image_id": iid,
                        "split": str(r.get("split", "train")).strip().lower(),
                        "landmark_id": str(r.get("landmark_id", "")).strip(),
                        "full_path": str(r.get("full_path", "")),
                        "relative_path": str(r.get("relative_path", "")),
                        "landmark_name": str(r.get("landmark_name", "")),
                        "is_resolved": bool(r.get("is_resolved", False)),
                        "category_url": r.get("category_url"),
                    }

        logger.info(
            "Loaded metadata for %d unique landmarks (%d total records) from %s",
            len(df_unique_landmarks),
            total_rows,
            parquet_path.name,
        )
        return total_rows

    def register_coordinates(
        self,
        split: str,
        landmark_id: str,
        latitude: float,
        longitude: float,
        source: str = "verified_external",
    ) -> None:
        """Register verified geographic coordinates into the distinct coordinates table."""
        if -90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0:
            key = (split.strip().lower(), str(landmark_id).strip())
            self._coordinates_table[key] = (latitude, longitude)
            logger.debug("Registered coordinates for %s: (%.5f, %.5f) [%s]", key, latitude, longitude, source)
        else:
            raise ValueError(f"Invalid coordinate bounds: ({latitude}, {longitude})")

    def lookup_landmark(self, split: str, landmark_id: str) -> LandmarkInfo:
        """Retrieve landmark details for a specific split and ID namespace."""
        key = (split.strip().lower(), str(landmark_id).strip())
        if key in self._landmark_cache:
            info = self._landmark_cache[key]
            # Attach verified coordinates if present
            if key in self._coordinates_table:
                lat, lon = self._coordinates_table[key]
                return LandmarkInfo(
                    split=info.split,
                    landmark_id=info.landmark_id,
                    landmark_name=info.landmark_name,
                    is_resolved=info.is_resolved,
                    category_url=info.category_url,
                    hierarchical_label=info.hierarchical_label,
                    supercategory=info.supercategory,
                    natural_or_human_made=info.natural_or_human_made,
                    latitude=lat,
                    longitude=lon,
                )
            return info

        # Fallback when landmark was not indexed
        return LandmarkResolver.resolve_landmark(split=split, landmark_id=landmark_id)

    def lookup_image(self, image_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve image record and resolved landmark info by image ID."""
        iid = str(image_id).strip()
        record = self._image_cache.get(iid)
        if record:
            key = (record["split"], record["landmark_id"])
            if key in self._coordinates_table:
                record["latitude"], record["longitude"] = self._coordinates_table[key]
            else:
                record["latitude"], record["longitude"] = None, None
        return record
