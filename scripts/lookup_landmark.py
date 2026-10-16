#!/usr/bin/env python3
"""Standalone Landmark Metadata & Geolocation Resolution CLI.

Resolves human-readable landmark names and hierarchy while strictly preserving
split-specific namespaces (train vs index). Coordinates are kept strictly distinct
from name identification and are never fabricated.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.metadata.lookup import LandmarkLookupTable
from src.metadata.resolver import LandmarkResolver
from src.utils.logging import get_logger

logger = get_logger("lookup_landmark_cli")


def main():
    parser = argparse.ArgumentParser(
        description="Lookup landmark name, category URL, and hierarchy with strict split preservation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", type=str, default=".env", help="Path to .env configuration file")
    parser.add_argument("--landmark-id", type=str, default=None, help="Landmark ID to resolve")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "index"],
        help="Dataset split namespace for landmark ID",
    )
    parser.add_argument("--image-id", type=str, default=None, help="Lookup landmark via image ID")
    parser.add_argument("--category-url", type=str, default=None, help="Directly test parsing a category URL")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    cfg = get_config(args.env_file)

    # Direct URL parser test mode
    if args.category_url:
        cleaned = LandmarkResolver.clean_category_url(args.category_url)
        res = {
            "input_url": args.category_url,
            "extracted_name": cleaned,
            "is_valid": cleaned is not None,
        }
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"Input URL      : {args.category_url}")
            print(f"Extracted Name : {cleaned}")
        return 0

    if not args.landmark_id and not args.image_id:
        logger.error("Please provide either --landmark-id, --image-id, or --category-url")
        return 1

    lookup = LandmarkLookupTable()
    # Attempt to load processed parquet metadata
    for split in ["train", "index"]:
        p = cfg.processed_dir / f"metadata_{split}.parquet"
        if p.is_file():
            lookup.load_from_parquet(p)

    if args.image_id:
        record = lookup.lookup_image(args.image_id)
        if not record:
            print(f"Image ID '{args.image_id}' not found in processed lookup table.")
            return 1
        split = record["split"]
        lid = record["landmark_id"]
        info = lookup.lookup_landmark(split, lid)
    else:
        split = args.split
        lid = args.landmark_id
        info = lookup.lookup_landmark(split, lid)

    result = {
        "split": info.split,
        "landmark_id": info.landmark_id,
        "landmark_name": info.landmark_name,
        "is_resolved": info.is_resolved,
        "category_url": info.category_url,
        "hierarchical_label": info.hierarchical_label,
        "supercategory": info.supercategory,
        "natural_or_human_made": info.natural_or_human_made,
        "latitude": info.latitude,
        "longitude": info.longitude,
        "coordinates_status": "Verified" if (info.latitude is not None and info.longitude is not None) else "Unknown / Not Provided in GLDv2 Metadata",
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print("\n" + "=" * 70)
    print(" LANDMARK METADATA RESOLUTION")
    print("=" * 70)
    print(f"  Landmark Name       : {result['landmark_name']}")
    print(f"  Resolution Status   : {'Successfully Resolved' if result['is_resolved'] else 'Unresolved (Fallback)'}")
    print(f"  Dataset Split       : {result['split']} (Preserved Namespace)")
    print(f"  Landmark ID         : {result['landmark_id']}")
    print(f"  Category Source     : {result['category_url'] or 'N/A'}")
    print(f"  Hierarchy / Label   : {result['hierarchical_label'] or 'N/A'}")
    print(f"  Supercategory       : {result['supercategory'] or 'N/A'}")
    print(f"  Class Type          : {result['natural_or_human_made'] or 'N/A'}")
    print(f"  Geographic Coords   : {result['coordinates_status']}")
    if result["latitude"] is not None:
        print(f"  Lat / Lon           : ({result['latitude']}, {result['longitude']})")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
