#!/usr/bin/env python3
"""Prepare, Clean, and Validate GLDv2 Metadata Artifacts.

Ingests official training and index metadata CSVs, verifies schemas and duplicate IDs,
resolves sharded directory structures, integrates category and hierarchical labels,
and exports clean Parquet tables.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.metadata.preparation import ensure_metadata_files, prepare_split_metadata
from src.utils.logging import get_logger

logger = get_logger("prepare_metadata_cli")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare, clean, and validate GLDv2 metadata into persistent Parquet artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", type=str, default=".env", help="Path to .env configuration file")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "index"],
        choices=["train", "index"],
        help="Dataset splits to process",
    )
    parser.add_argument("--sample-limit", type=int, default=None, help="Process only the first N records for testing")
    parser.add_argument(
        "--verify-images",
        action="store_true",
        default=True,
        help="Check whether resolved image files exist on disk",
    )
    parser.add_argument(
        "--no-verify-images",
        dest="verify_images",
        action="store_false",
        help="Skip checking if image files exist on disk (fast metadata-only mode)",
    )
    parser.add_argument(
        "--download-metadata",
        action="store_true",
        help="Download missing official metadata CSVs if absent",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if output parquet file exists",
    )
    args = parser.parse_args()

    cfg = get_config(args.env_file)
    cfg.ensure_directories()

    auto_dl = args.download_metadata or cfg.auto_download_metadata

    # Check and optionally download metadata files
    ensure_metadata_files(
        metadata_dir=cfg.metadata_dir,
        splits=args.splits,
        auto_download=auto_dl,
    )

    summaries = []
    for split in args.splits:
        out_parquet = cfg.processed_dir / f"metadata_{split}.parquet"
        if out_parquet.is_file() and not args.force and args.sample_limit is None:
            logger.info("Found existing validated metadata for '%s' at %s (use --force to overwrite)", split, out_parquet)
            continue

        try:
            summary = prepare_split_metadata(
                split=split,
                metadata_dir=cfg.metadata_dir,
                image_dir=cfg.image_dir,
                output_parquet=out_parquet,
                sample_limit=args.sample_limit,
                verify_images=args.verify_images,
            )
            summaries.append(summary)
        except Exception as e:
            logger.error("Failed to prepare metadata for split '%s': %s", split, e)
            return 1

    print("\n" + "=" * 70)
    print(" METADATA PREPARATION SUMMARY")
    print("=" * 70)
    for s in summaries:
        print(f"Split: {s['split'].upper()}")
        print(f"  Total records processed : {s['total_records']}")
        print(f"  Duplicate IDs removed   : {s['duplicates_removed']}")
        print(f"  Images verified on disk : {s['existing_images']}")
        print(f"  Missing image files     : {s['missing_images']}")
        print(f"  Resolved landmark names : {s['resolved_landmark_names']}")
        print(f"  Unique landmarks        : {s['unique_landmarks']}")
        print(f"  Artifact saved to       : {s['output_parquet']}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
