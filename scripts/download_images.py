#!/usr/bin/env python3
"""GLDv2 Image Downloader Utility.

Supports two download modes:
1. Fast URL Downloader (--from-csv): Downloads a configurable number of real images
   concurrently directly from the URLs in train.csv into the official sharded directory format.
2. Official Tar Chunk Downloader (--from-tar): Downloads and extracts official GLDv2
   packaged tarballs (images_000.tar, etc.) from the CVD Foundation S3 bucket.
"""

import argparse
import concurrent.futures
import io
import os
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.metadata.preparation import resolve_image_path
from src.utils.logging import get_logger

logger = get_logger("download_images")

# Official CVD Foundation tar URL pattern
# Total 500 tar chunks for train split: images_000.tar to images_499.tar (~1GB each)
TAR_URL_PATTERN = "https://s3.amazonaws.com/google-landmark/train/images_{chunk_id:03d}.tar"


def download_single_image(row: dict, image_dir: Path) -> Tuple[str, bool, str]:
    """Download a single image from URL and save to GLDv2 sharded path."""
    img_id = str(row["id"]).strip()
    url = str(row["url"]).strip()
    split = str(row.get("split", "train")).strip().lower()

    rel_path, abs_path = resolve_image_path(split, img_id, image_dir)
    dest_path = Path(abs_path)

    if dest_path.is_file() and dest_path.stat().st_size > 0:
        return img_id, True, "already_exists"

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "GLDv2-Academic-Research-Search-Engine/1.0 (Educational/Research ML Pipeline)"
    }
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = response.read()

        # Validate that bytes represent a valid readable image before saving
        with Image.open(io.BytesIO(data)) as img:
            img.verify()

        with open(dest_path, "wb") as f:
            f.write(data)

        return img_id, True, "success"
    except Exception as e:
        return img_id, False, str(e)


def download_images_from_csv(
    metadata_csv: Path,
    image_dir: Path,
    max_images: int = 1000,
    num_threads: int = 16,
    split: str = "train",
) -> None:
    """Download a targeted batch of images directly from the URLs in train.csv."""
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    logger.info("Reading image URLs from %s...", metadata_csv)
    df = pd.read_csv(metadata_csv, dtype={"id": str, "url": str, "landmark_id": str})
    df["split"] = split

    if "url" not in df.columns or "id" not in df.columns:
        raise ValueError(f"CSV missing 'id' or 'url' columns. Found: {df.columns.tolist()}")

    # Filter out already existing images
    pending_records = []
    for _, row in df.iterrows():
        _, abs_path = resolve_image_path(split, row["id"], image_dir)
        if not os.path.isfile(abs_path):
            pending_records.append(row.to_dict())
        if len(pending_records) >= max_images:
            break

    logger.info("Target download count: %d images using %d threads...", len(pending_records), num_threads)

    if not pending_records:
        logger.info("All requested images are already present on disk.")
        return

    success_count = 0
    fail_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(download_single_image, record, image_dir): record
            for record in pending_records
        }

        pbar = tqdm(total=len(pending_records), desc="Downloading Images")
        for future in concurrent.futures.as_completed(futures):
            _, success, msg = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
            pbar.update(1)
        pbar.close()

    logger.info(
        "Download complete. Successfully downloaded: %d images | Failed / Corrupt: %d",
        success_count,
        fail_count,
    )


def download_official_tar_chunks(
    chunk_start: int,
    chunk_end: int,
    image_dir: Path,
    temp_dir: Path,
) -> None:
    """Download and extract official packaged tar files (each contains ~8,000 images)."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    target_extract = image_dir / "train"
    target_extract.mkdir(parents=True, exist_ok=True)

    for chunk_id in range(chunk_start, chunk_end + 1):
        url = TAR_URL_PATTERN.format(chunk_id=chunk_id)
        tar_filename = f"images_{chunk_id:03d}.tar"
        dest_tar = temp_dir / tar_filename

        logger.info("Downloading official chunk [%d/%d]: %s", chunk_id, chunk_end, url)
        try:
            # Download with curl / urllib
            with urllib.request.urlopen(url) as resp, open(dest_tar, "wb") as f_out:
                total = int(resp.headers.get("content-length", 0))
                pbar = tqdm(total=total, unit="B", unit_scale=True, desc=tar_filename)
                while True:
                    buf = resp.read(1024 * 64)
                    if not buf:
                        break
                    f_out.write(buf)
                    pbar.update(len(buf))
                pbar.close()

            logger.info("Extracting %s into %s...", tar_filename, target_extract)
            with tarfile.open(dest_tar, "r") as tar:
                tar.extractall(path=target_extract)

            # Clean up tar file to save disk space
            dest_tar.unlink()
            logger.info("Successfully processed and extracted chunk %d.", chunk_id)
        except Exception as e:
            logger.error("Failed to download or extract tar chunk %d: %s", chunk_id, e)


def main():
    parser = argparse.ArgumentParser(
        description="Download real GLDv2 images from CSV URLs or official CVD Foundation tarballs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", type=str, default=".env", help="Path to .env configuration file")
    parser.add_argument("--mode", choices=["from-csv", "from-tar"], default="from-csv", help="Download method")
    parser.add_argument("--max-images", type=int, default=1000, help="Max images to download in --from-csv mode")
    parser.add_argument("--num-threads", type=int, default=16, help="Worker threads for concurrent HTTP downloads")
    parser.add_argument("--tar-start", type=int, default=0, help="First tar chunk index (0 to 499)")
    parser.add_argument("--tar-end", type=int, default=0, help="Last tar chunk index (0 to 499)")
    args = parser.parse_args()

    cfg = get_config(args.env_file)
    cfg.ensure_directories()

    if args.mode == "from-csv":
        csv_file = cfg.metadata_dir / "train.csv"
        download_images_from_csv(
            metadata_csv=csv_file,
            image_dir=cfg.image_dir,
            max_images=args.max_images,
            num_threads=args.num_threads,
            split="train",
        )
    elif args.mode == "from-tar":
        download_official_tar_chunks(
            chunk_start=args.tar_start,
            chunk_end=args.tar_end,
            image_dir=cfg.image_dir,
            temp_dir=cfg.data_root / "temp_tars",
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
