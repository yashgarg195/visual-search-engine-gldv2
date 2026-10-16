"""Landmark Name Resolution, URL Parsing, and Namespace Management.

Strictly preserves split-specific namespaces (train vs index), extracts human-readable
names from Wikimedia Commons URLs, and falls back to hierarchical metadata.
Never fabricates landmark names or geographic coordinates.
"""

import re
import urllib.parse
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LandmarkInfo:
    """Structured landmark metadata representation."""

    split: str
    landmark_id: str
    landmark_name: str
    is_resolved: bool
    category_url: Optional[str] = None
    hierarchical_label: Optional[str] = None
    supercategory: Optional[str] = None
    natural_or_human_made: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class LandmarkResolver:
    """Extracts human-readable landmark names and resolves metadata cleanly."""

    # Matches URL prefix for Wikimedia Commons categories
    _WIKI_CATEGORY_PATTERN = re.compile(
        r"^(?:https?://commons\.wikimedia\.org/wiki/)?(?:Category:)?(.+)$",
        re.IGNORECASE,
    )

    @classmethod
    def clean_category_url(cls, raw_url: Optional[str]) -> Optional[str]:
        """Convert a Wikimedia Commons category URL into a clean, human-readable name.

        Examples:
            'http://commons.wikimedia.org/wiki/Category:Taj_Mahal' -> 'Taj Mahal'
            'http://commons.wikimedia.org/wiki/Category:%C3%89cole_Militaire' -> 'École Militaire'
            'http://commons.wikimedia.org/wiki/Category:Notre-Dame_de_Paris' -> 'Notre-Dame de Paris'
            'Category:Eiffel_Tower' -> 'Eiffel Tower'
            'http://commons.wikimedia.org/wiki/Category:St._Peter%27s_Basilica' -> "St. Peter's Basilica"
        """
        if not raw_url or not isinstance(raw_url, str):
            return None

        url_str = raw_url.strip()
        if not url_str or url_str.lower() in ("none", "null", "nan"):
            return None

        # URL decode percent-encoded characters (e.g. %20 -> space, %C3%89 -> É)
        decoded = urllib.parse.unquote(url_str)

        match = cls._WIKI_CATEGORY_PATTERN.match(decoded)
        if match:
            extracted = match.group(1).strip()
        else:
            extracted = decoded.strip()

        # Replace underscores with spaces
        name = extracted.replace("_", " ")

        # Clean up any consecutive whitespace
        name = re.sub(r"\s+", " ", name).strip()

        return name if name else None

    @classmethod
    def resolve_landmark(
        cls,
        split: str,
        landmark_id: str,
        category_url: Optional[str] = None,
        hierarchical_label: Optional[str] = None,
        supercategory: Optional[str] = None,
        natural_or_human_made: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> LandmarkInfo:
        """Resolve landmark details while preserving split namespaces and verifying resolution."""
        split_clean = str(split).strip().lower()
        id_clean = str(landmark_id).strip()

        extracted_name = cls.clean_category_url(category_url)
        is_resolved = True

        if extracted_name:
            final_name = extracted_name
        elif hierarchical_label and str(hierarchical_label).strip() not in ("", "none", "null", "nan"):
            # Fallback 1: Hierarchical label if available
            label_clean = str(hierarchical_label).strip().replace("_", " ")
            final_name = f"{label_clean}"
        elif supercategory and str(supercategory).strip() not in ("", "none", "null", "nan"):
            # Fallback 2: Supercategory if available
            final_name = f"{supercategory.strip().replace('_', ' ')} (Category #{id_clean})"
        else:
            # Unresolved landmark: Never fabricate a fake name
            final_name = f"Unknown Landmark ({split_clean} ID: {id_clean})"
            is_resolved = False

        # Validate coordinate bounds if provided
        valid_lat = latitude if (latitude is not None and -90.0 <= latitude <= 90.0) else None
        valid_lon = longitude if (longitude is not None and -180.0 <= longitude <= 180.0) else None

        return LandmarkInfo(
            split=split_clean,
            landmark_id=id_clean,
            landmark_name=final_name,
            is_resolved=is_resolved,
            category_url=category_url if category_url and category_url.strip() else None,
            hierarchical_label=hierarchical_label if hierarchical_label else None,
            supercategory=supercategory if supercategory else None,
            natural_or_human_made=natural_or_human_made if natural_or_human_made else None,
            latitude=valid_lat,
            longitude=valid_lon,
        )
