"""Metadata Preparation, Ingestion, and Path Resolution for GLDv2.

Handles training and index splits, validates schemas and duplicate IDs,
resolves sharded directory structures, integrates category/hierarchical labels,
and exports clean, persistent Parquet artifacts.
"""

import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from tqdm import tqdm

from src.metadata.resolver import LandmarkResolver
from src.utils.io import atomic_save_json, atomic_save_parquet
from src.utils.logging import get_logger

logger = get_logger("metadata_preparation")

# Official GLDv2 metadata URLs
METADATA_URLS = {
    "train_metadata": "https://s3.amazonaws.com/google-landmark/metadata/train.csv",
    "train_category": "https://s3.amazonaws.com/google-landmark/metadata/train_label_to_category.csv",
    "train_hierarchical": "https://s3.amazonaws.com/google-landmark/metadata/train_label_to_hierarchical.csv",
    "index_images": "https://s3.amazonaws.com/google-landmark/metadata/index.csv",
    "index_landmark_map": "https://s3.amazonaws.com/google-landmark/metadata/index_image_to_landmark.csv",
    "index_category": "https://s3.amazonaws.com/google-landmark/metadata/index_label_to_category.csv",
    "index_hierarchical": "https://s3.amazonaws.com/google-landmark/metadata/index_label_to_hierarchical.csv",
}


def download_file_with_progress(url: str, dest_path: Path) -> None:
    """Download a file with a progress bar and atomic temporary file destination."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(f"{dest_path.suffix}.tmp.{os.getpid()}")

    logger.info("Downloading %s -> %s", url, dest_path)
    try:
        with urllib.request.urlopen(url) as response, open(temp_path, "wb") as out_file:
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024 * 64
            pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc=dest_path.name)
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                out_file.write(buffer)
                pbar.update(len(buffer))
            pbar.close()
        os.replace(temp_path, dest_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def ensure_metadata_files(
    metadata_dir: Path,
    splits: List[str],
    auto_download: bool = False,
) -> Dict[str, Path]:
    """Verify presence of metadata files for specified splits, downloading if permitted."""
    required_keys = []
    if "train" in splits:
        required_keys.extend(["train_metadata", "train_category", "train_hierarchical"])
    if "index" in splits:
        required_keys.extend(["index_images", "index_landmark_map", "index_category", "index_hierarchical"])

    resolved_paths: Dict[str, Path] = {}
    missing_files: List[Tuple[str, str, Path]] = []

    for key in required_keys:
        filename = METADATA_URLS[key].split("/")[-1]
        target_path = metadata_dir / filename
        resolved_paths[key] = target_path
        if not target_path.is_file():
            missing_files.append((key, METADATA_URLS[key], target_path))

    if missing_files:
        if auto_download:
            logger.info("Downloading %d missing metadata files...", len(missing_files))
            for key, url, path in missing_files:
                download_file_with_progress(url, path)
        else:
            msg = (
                f"Missing {len(missing_files)} required metadata files in {metadata_dir}:\n"
                + "\n".join([f"  - {p.name} (Source: {u})" for _, u, p in missing_files])
                + "\nEnable auto-download with --download-metadata or place files manually."
            )
            logger.warning(msg)

    return resolved_paths


def resolve_image_path(split: str, image_id: str, image_dir: Path) -> Tuple[str, str]:
    """Calculate the relative and absolute sharded path for an image ID.

    GLDv2 format: {split}/{char0}/{char1}/{char2}/{image_id}.jpg
    """
    id_str = str(image_id).strip()
    c0 = id_str[0] if len(id_str) > 0 else "_"
    c1 = id_str[1] if len(id_str) > 1 else "_"
    c2 = id_str[2] if len(id_str) > 2 else "_"

    rel_path = f"{split}/{c0}/{c1}/{c2}/{id_str}.jpg"
    abs_path = str(image_dir / rel_path)
    return rel_path, abs_path


def prepare_split_metadata(
    split: str,
    metadata_dir: Path,
    image_dir: Path,
    output_parquet: Path,
    sample_limit: Optional[int] = None,
    verify_images: bool = True,
) -> Dict[str, Any]:
    """Process, clean, and validate metadata for a single dataset split (train or index)."""
    split = split.lower().strip()
    if split not in ("train", "index"):
        raise ValueError(f"Invalid split '{split}'. Expected 'train' or 'index'.")

    logger.info("Preparing metadata for split: '%s'", split)

    # 1. Load primary image mapping
    if split == "train":
        meta_file = metadata_dir / "train.csv"
        if not meta_file.is_file():
            raise FileNotFoundError(f"Training metadata file not found: {meta_file}")

        # String dtype preserves leading zeros
        df_images = pd.read_csv(
            meta_file,
            dtype={"id": str, "landmark_id": str},
            nrows=sample_limit,
        )
        if "id" not in df_images.columns or "landmark_id" not in df_images.columns:
            raise ValueError(f"train.csv missing required columns: {df_images.columns.tolist()}")

    else:  # index split
        index_file = metadata_dir / "index.csv"
        map_file = metadata_dir / "index_image_to_landmark.csv"
        if not index_file.is_file():
            raise FileNotFoundError(f"Index images file not found: {index_file}")

        df_images = pd.read_csv(index_file, dtype={"id": str}, nrows=sample_limit)
        if map_file.is_file():
            df_map = pd.read_csv(map_file, dtype={"id": str, "landmark_id": str})
            df_images = df_images.merge(df_map, on="id", how="left")
        else:
            df_images["landmark_id"] = "unknown"

    initial_count = len(df_images)
    logger.info("Loaded %d raw image records for split '%s'.", initial_count, split)

    # 2. Check and report duplicate IDs
    duplicates_mask = df_images.duplicated(subset=["id"], keep="first")
    duplicate_count = int(duplicates_mask.sum())
    if duplicate_count > 0:
        logger.warning("Found and removed %d duplicate image IDs in '%s'.", duplicate_count, split)
        df_images = df_images[~duplicates_mask].copy()

    # 3. Load category mappings
    cat_file = metadata_dir / f"{split}_label_to_category.csv"
    category_map: Dict[str, str] = {}
    if cat_file.is_file():
        df_cat = pd.read_csv(cat_file, dtype={"landmark_id": str, "category": str})
        category_map = dict(zip(df_cat["landmark_id"].astype(str), df_cat["category"].astype(str)))
        logger.info("Loaded %d category mappings from %s", len(category_map), cat_file.name)
    else:
        logger.warning("Category file %s not found. Landmark names will be extracted from fallback.", cat_file.name)

    # 4. Load hierarchical mappings
    hier_file = metadata_dir / f"{split}_label_to_hierarchical.csv"
    hier_map: Dict[str, Dict[str, str]] = {}
    if hier_file.is_file():
        df_hier = pd.read_csv(hier_file, dtype=str)
        if "landmark_id" in df_hier.columns:
            for _, row in df_hier.iterrows():
                hier_map[str(row["landmark_id"])] = {
                    "hierarchical_label": str(row.get("hierarchical_label", "")),
                    "supercategory": str(row.get("supercategory", "")),
                    "natural_or_human_made": str(row.get("natural_or_human_made", "")),
                }
        logger.info("Loaded %d hierarchical mappings from %s", len(hier_map), hier_file.name)

    # 5. Build cleaned, resolved records
    rows = []
    missing_images_count = 0
    resolved_names_count = 0

    logger.info("Resolving image paths and landmark names...")
    for _, row in tqdm(df_images.iterrows(), total=len(df_images), desc=f"Processing {split}"):
        img_id = str(row["id"]).strip()
        raw_landmark_id = str(row.get("landmark_id", "")).strip()

        rel_path, abs_path = resolve_image_path(split, img_id, image_dir)
        exists = os.path.isfile(abs_path) if verify_images else True
        if not exists:
            missing_images_count += 1

        cat_url = category_map.get(raw_landmark_id)
        hier_info = hier_map.get(raw_landmark_id, {})

        landmark_info = LandmarkResolver.resolve_landmark(
            split=split,
            landmark_id=raw_landmark_id,
            category_url=cat_url,
            hierarchical_label=hier_info.get("hierarchical_label"),
            supercategory=hier_info.get("supercategory"),
            natural_or_human_made=hier_info.get("natural_or_human_made"),
        )

        if landmark_info.is_resolved:
            resolved_names_count += 1

        rows.append({
            "image_id": img_id,
            "split": split,
            "landmark_id": raw_landmark_id,
            "relative_path": rel_path,
            "full_path": abs_path,
            "exists": exists,
            "landmark_name": landmark_info.landmark_name,
            "is_resolved": landmark_info.is_resolved,
            "category_url": landmark_info.category_url,
            "hierarchical_label": landmark_info.hierarchical_label,
            "supercategory": landmark_info.supercategory,
            "natural_or_human_made": landmark_info.natural_or_human_made,
        })

    df_clean = pd.DataFrame(rows)

    # 6. Save persistent parquet artifact
    atomic_save_parquet(df_clean, output_parquet)
    logger.info("Saved cleaned metadata (%d records) to %s", len(df_clean), output_parquet)

    summary = {
        "split": split,
        "total_records": len(df_clean),
        "duplicates_removed": duplicate_count,
        "existing_images": int((df_clean["exists"] == True).sum()),
        "missing_images": missing_images_count,
        "resolved_landmark_names": resolved_names_count,
        "unique_landmarks": int(df_clean["landmark_id"].nunique()),
        "output_parquet": str(output_parquet),
    }

    # Save summary json alongside parquet
    summary_path = output_parquet.with_suffix(".summary.json")
    atomic_save_json(summary, summary_path)
    return summary
