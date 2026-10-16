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
        loaded = 0

        for _, row in df.iterrows():
            split = str(row.get("split", "train")).strip().lower()
            lid = str(row.get("landmark_id", "")).strip()
            iid = str(row.get("image_id", "")).strip()

            key = (split, lid)
            if key not in self._landmark_cache:
                info = LandmarkInfo(
                    split=split,
                    landmark_id=lid,
                    landmark_name=str(row.get("landmark_name", f"Landmark #{lid}")),
                    is_resolved=bool(row.get("is_resolved", False)),
                    category_url=row.get("category_url"),
                    hierarchical_label=row.get("hierarchical_label"),
                    supercategory=row.get("supercategory"),
                    natural_or_human_made=row.get("natural_or_human_made"),
                )
                self._landmark_cache[key] = info

            if iid:
                self._image_cache[iid] = {
                    "image_id": iid,
                    "split": split,
                    "landmark_id": lid,
                    "full_path": str(row.get("full_path", "")),
                    "relative_path": str(row.get("relative_path", "")),
                    "landmark_name": str(row.get("landmark_name", "")),
                    "is_resolved": bool(row.get("is_resolved", False)),
                    "category_url": row.get("category_url"),
                }
            loaded += 1

        logger.info("Loaded %d records into lookup table from %s", loaded, parquet_path.name)
        return loaded

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
