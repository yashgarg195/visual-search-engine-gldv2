"""Metadata processing, namespace preservation, and landmark name resolution."""

from src.metadata.lookup import LandmarkLookupTable
from src.metadata.preparation import (
    ensure_metadata_files,
    prepare_split_metadata,
    resolve_image_path,
)
from src.metadata.resolver import LandmarkInfo, LandmarkResolver

__all__ = [
    "LandmarkInfo",
    "LandmarkResolver",
    "LandmarkLookupTable",
    "ensure_metadata_files",
    "prepare_split_metadata",
    "resolve_image_path",
]
