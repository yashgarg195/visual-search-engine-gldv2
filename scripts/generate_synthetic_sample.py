#!/usr/bin/env python3
"""Synthetic Sample Generator for GLDv2 Local Testing.

Generates a realistic small sample of images and authentic GLDv2 metadata CSVs
following the exact GLDv2 sharded directory structure:
    {split}/{char0}/{char1}/{char2}/{id}.jpg

Enables immediate end-to-end local testing without downloading gigabytes of data.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd
from PIL import Image, ImageDraw

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.utils.logging import get_logger

logger = get_logger("sample_generator")

# Representative authentic landmarks with category URLs and hierarchies
SAMPLE_LANDMARKS = [
    {
        "id": "1001",
        "name": "Eiffel Tower",
        "category": "http://commons.wikimedia.org/wiki/Category:Eiffel_Tower",
        "hierarchical_label": "monument_tower",
        "supercategory": "architecture",
        "natural_or_human_made": "human-made",
        "color": (210, 180, 140),
    },
    {
        "id": "1002",
        "name": "Taj Mahal",
        "category": "http://commons.wikimedia.org/wiki/Category:Taj_Mahal",
        "hierarchical_label": "mausoleum_mosque",
        "supercategory": "heritage_site",
        "natural_or_human_made": "human-made",
        "color": (245, 245, 245),
    },
    {
        "id": "1003",
        "name": "Colosseum",
        "category": "http://commons.wikimedia.org/wiki/Category:Colosseum",
        "hierarchical_label": "amphitheatre_ruins",
        "supercategory": "ancient_ruins",
        "natural_or_human_made": "human-made",
        "color": (195, 160, 120),
    },
    {
        "id": "1004",
        "name": "Big Ben",
        "category": "http://commons.wikimedia.org/wiki/Category:Big_Ben_(London)",
        "hierarchical_label": "clock_tower",
        "supercategory": "architecture",
        "natural_or_human_made": "human-made",
        "color": (160, 150, 140),
    },
    {
        "id": "1005",
        "name": "Mount Fuji",
        "category": "http://commons.wikimedia.org/wiki/Category:Mount_Fuji",
        "hierarchical_label": "volcano_mountain",
        "supercategory": "natural_feature",
        "natural_or_human_made": "natural",
        "color": (100, 140, 180),
    },
]


def create_sample_image(
    dest_path: Path,
    base_color: tuple,
    label_text: str,
    img_id: str,
    variation: int,
) -> None:
    """Generate a clean synthetic JPEG image with geometric patterns and labels."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (256, 256), color=base_color)
    draw = ImageDraw.Draw(img)

    # Draw distinctive geometric variations for feature extraction
    offset = (variation % 5) * 10
    draw.rectangle([30 + offset, 40 + offset, 220 - offset, 210 - offset], outline=(50, 50, 50), width=4)
    draw.line([20, 20, 236, 236], fill=(70, 70, 70), width=2)
    draw.ellipse([80 + offset, 80 + offset, 170 - offset, 170 - offset], fill=(230, 200, 150), outline=(30, 30, 30))

    # Add text label
    draw.text((10, 10), f"ID: {img_id}", fill=(20, 20, 20))
    draw.text((10, 230), label_text[:25], fill=(20, 20, 20))

    img.save(dest_path, "JPEG", quality=90)


def generate_dataset_sample(
    data_dir: Path,
    images_per_landmark: int = 12,
) -> None:
    """Create a self-contained GLDv2 sample directory with images and metadata CSVs."""
    image_root = data_dir / "raw" / "images"
    meta_root = data_dir / "raw" / "metadata"

    image_root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)

    train_rows = []
    index_rows = []
    index_map_rows = []
    cat_rows = []
    hier_rows = []

    img_counter = 1

    for landmark in SAMPLE_LANDMARKS:
        lid = landmark["id"]
        lname = landmark["name"]
        color = landmark["color"]

        # Category and hierarchy rows
        cat_rows.append({"landmark_id": lid, "category": landmark["category"]})
        hier_rows.append({
            "landmark_id": lid,
            "hierarchical_label": landmark["hierarchical_label"],
            "supercategory": landmark["supercategory"],
            "natural_or_human_made": landmark["natural_or_human_made"],
        })

        # Generate images for train split
        for v in range(images_per_landmark):
            # Hexadecimal string ID with leading zeros (e.g. 0001a1, 0001a2)
            img_id = f"{img_counter:06x}"
            img_counter += 1

            # Path format: {split}/{c0}/{c1}/{c2}/{img_id}.jpg
            c0, c1, c2 = img_id[0], img_id[1], img_id[2]
            img_path = image_root / "train" / c0 / c1 / c2 / f"{img_id}.jpg"
            create_sample_image(img_path, color, lname, img_id, v)

            train_rows.append({
                "id": img_id,
                "url": f"https://example.com/images/{img_id}.jpg",
                "landmark_id": lid,
            })

        # Generate a couple of images for index split
        for v in range(3):
            img_id = f"{img_counter:06x}"
            img_counter += 1

            c0, c1, c2 = img_id[0], img_id[1], img_id[2]
            img_path = image_root / "index" / c0 / c1 / c2 / f"{img_id}.jpg"
            create_sample_image(img_path, color, lname, img_id, v + 20)

            index_rows.append({
                "id": img_id,
                "url": f"https://example.com/images/{img_id}.jpg",
            })
            index_map_rows.append({
                "id": img_id,
                "landmark_id": lid,
            })

    # Save official CSV metadata files
    pd.DataFrame(train_rows).to_csv(meta_root / "train.csv", index=False)
    pd.DataFrame(cat_rows).to_csv(meta_root / "train_label_to_category.csv", index=False)
    pd.DataFrame(hier_rows).to_csv(meta_root / "train_label_to_hierarchical.csv", index=False)

    pd.DataFrame(index_rows).to_csv(meta_root / "index.csv", index=False)
    pd.DataFrame(index_map_rows).to_csv(meta_root / "index_image_to_landmark.csv", index=False)
    pd.DataFrame(cat_rows).to_csv(meta_root / "index_label_to_category.csv", index=False)
    pd.DataFrame(hier_rows).to_csv(meta_root / "index_label_to_hierarchical.csv", index=False)

    # Create one extra standalone query image for search demonstrations
    query_dir = data_dir / "sample_queries"
    query_dir.mkdir(parents=True, exist_ok=True)
    query_path = query_dir / "query_eiffel_tower.jpg"
    create_sample_image(query_path, SAMPLE_LANDMARKS[0]["color"], "Eiffel Tower (Query)", "query01", 99)

    logger.info("Successfully generated sample dataset at %s:", data_dir)
    logger.info("  - Train images : %d", len(train_rows))
    logger.info("  - Index images : %d", len(index_rows))
    logger.info("  - Landmarks    : %d", len(SAMPLE_LANDMARKS))
    logger.info("  - Query test   : %s", query_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a small synthetic GLDv2 sample dataset for local offline testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", type=str, default=".env", help="Path to .env configuration file")
    parser.add_argument("--images-per-landmark", type=int, default=12, help="Number of train images per landmark")
    args = parser.parse_args()

    cfg = get_config(args.env_file)
    generate_dataset_sample(cfg.data_root, images_per_landmark=args.images_per_landmark)
    return 0


if __name__ == "__main__":
    sys.exit(main())
