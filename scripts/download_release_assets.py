#!/usr/bin/env python3
"""Utility script to download indexed FAISS models and metadata tables from GitHub Releases.

Usage:
    python scripts/download_release_assets.py [--tag v1.0.0] [--output-dir outputs/index]
"""

import argparse
import sys
import urllib.request
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import get_logger

logger = get_logger("release_downloader")

DEFAULT_REPO = "yashgarg195/visual-search-engine-gldv2"
DEFAULT_TAG = "v1.0.0"

ASSETS = [
    {
        "filename": "ivfpq_populated.index",
        "description": "5.05-million image populated FAISS IVF-PQ index (394 MB)",
    },
    {
        "filename": "id_mapping.parquet",
        "description": "ID mapping table mapping FAISS integer IDs to image & landmark IDs (122 MB)",
    },
    {
        "filename": "ivfpq_trained.index",
        "description": "Pre-trained empty IVF-PQ index centroids (8.5 MB)",
    },
]


def download_file(url: str, dest_path: Path):
    """Download a file with a progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    logger.info("Downloading %s -> %s...", url, dest_path.name)

    def report_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {percent:5.1f}% [{mb_down:6.1f} / {mb_total:6.1f} MB]")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, temp_path, reporthook=report_hook)
        print()  # Newline
        temp_path.replace(dest_path)
        logger.info("Successfully downloaded %s (%d bytes).", dest_path.name, dest_path.stat().st_size)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed downloading {url}: {e}") from e


def main():
    parser = argparse.ArgumentParser(description="Download GLDv2 index and metadata assets from GitHub Releases.")
    parser.add_argument("--repo", type=str, default=DEFAULT_REPO, help="GitHub repository (owner/repo)")
    parser.add_argument("--tag", type=str, default=DEFAULT_TAG, help="Release tag name")
    parser.add_argument("--output-dir", type=str, default="outputs/index", help="Output directory for index assets")
    parser.add_argument("--force", action="store_true", help="Overwrite existing assets if present")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"https://github.com/{args.repo}/releases/download/{args.tag}"

    logger.info("Checking release assets from %s (%s)...", args.repo, args.tag)

    for item in ASSETS:
        fname = item["filename"]
        target = out_dir / fname
        if target.is_file() and not args.force:
            logger.info("Asset '%s' already exists (%d bytes). Skipping. (Use --force to re-download)", fname, target.stat().st_size)
            continue

        url = f"{base_url}/{fname}"
        download_file(url, target)

    logger.info("All requested release assets are verified and ready in %s!", out_dir)


if __name__ == "__main__":
    main()
