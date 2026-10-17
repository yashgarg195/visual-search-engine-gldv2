#!/usr/bin/env python3
"""Pipelined Streaming Downloader & Batch Embedder for GLDv2.

Downloads GLDv2 official tar chunks in batches, immediately extracts them,
runs OpenCLIP batch embedding on the GPU, writes atomic embedding shards,
and can optionally delete processed raw images to save disk space if configured.
Guarantees resumability and prevents duplicate downloads.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import List, Set
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.metadata.preparation import ensure_metadata_files, prepare_split_metadata
from src.models.clip_encoder import CLIPEncoder
from src.storage.embedding_store import EmbeddingStore
from src.utils.logging import get_logger

logger = get_logger("stream_pipeline")

import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, Future

def download_chunk(split: str, chunk_id: int, dest_tar: Path) -> bool:
    """Download single tar chunk using aria2c if available (multi-connection), falling back to urllib."""
    url = f"https://s3.amazonaws.com/google-landmark/{split}/images_{chunk_id:03d}.tar"
    temp_tar = dest_tar.with_suffix(f".tmp.{os.getpid()}")

    # Try aria2c first if installed
    if shutil.which("aria2c"):
        cmd = [
            "aria2c",
            "-x", "8",
            "-s", "8",
            "-j", "8",
            "-k", "1M",
            "--summary-interval=0",
            "--console-log-level=warn",
            "-d", str(temp_tar.parent),
            "-o", temp_tar.name,
            url,
        ]
        ret = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if ret.returncode == 0 and temp_tar.exists():
            os.replace(temp_tar, dest_tar)
            return True
        logger.warning("aria2c download failed (code %d): %s. Falling back to urllib.", ret.returncode, ret.stderr)

    # Fallback to urllib
    try:
        with urllib.request.urlopen(url) as resp, open(temp_tar, "wb") as f_out:
            total = int(resp.headers.get("content-length", 0))
            pbar = tqdm(total=total, unit="B", unit_scale=True, desc=f"{split} Chunk {chunk_id:03d}.tar", leave=False)
            while True:
                buf = resp.read(1024 * 64)
                if not buf:
                    break
                f_out.write(buf)
                pbar.update(len(buf))
            pbar.close()
        os.replace(temp_tar, dest_tar)
        return True
    except Exception as e:
        logger.error("Failed downloading %s chunk %d: %s", split, chunk_id, e)
        if temp_tar.exists():
            temp_tar.unlink()
        return False


def extract_tar(tar_path: Path, target_dir: Path) -> None:
    """Extract tar using fast system tar binary if available, falling back to Python tarfile."""
    logger.info("Extracting %s into %s...", tar_path.name, target_dir)
    if shutil.which("tar"):
        ret = subprocess.run(["tar", "-C", str(target_dir), "-xf", str(tar_path)], capture_output=True)
        if ret.returncode == 0:
            tar_path.unlink()
            return
        logger.warning("System tar failed (code %d): %s. Falling back to tarfile.", ret.returncode, ret.stderr)

    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=target_dir)
    tar_path.unlink()


def main():
    parser = argparse.ArgumentParser(description="Pipelined Tar Downloader and GPU Vectorizer")
    parser.add_argument("--split", type=str, default="train", choices=["train", "index", "test"], help="Dataset split to process")
    parser.add_argument("--start-chunk", type=int, default=0, help="Starting tar chunk")
    parser.add_argument("--end-chunk", type=int, default=None, help="Ending tar chunk (default: 499 for train, 99 for index, 19 for test)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for GPU inference")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of CPU worker threads for image decoding")
    parser.add_argument("--shard-size", type=int, default=10000, help="Images per persistent embedding shard")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    parser.add_argument("--cleanup-images-after-embed", action="store_true", help="Delete images after embedding to save disk")
    args = parser.parse_args()

    cfg = get_config()
    cfg.ensure_directories()

    split = args.split
    max_chunks = {"train": 499, "index": 99, "test": 19}
    end_chunk = args.end_chunk if args.end_chunk is not None else max_chunks[split]

    metadata_parquet = cfg.processed_dir / f"metadata_{split}.parquet"

    # Step 1: Ensure metadata exists
    if not metadata_parquet.is_file():
        logger.info("Downloading and preparing GLDv2 %s metadata...", split)
        ensure_metadata_files(cfg.metadata_dir, splits=[split], auto_download=True)
        prepare_split_metadata(
            split=split,
            metadata_dir=cfg.metadata_dir,
            image_dir=cfg.image_dir,
            output_parquet=metadata_parquet,
            verify_images=False,
        )

    logger.info("Loading metadata table from %s...", metadata_parquet)
    df_meta = pd.read_parquet(metadata_parquet)
    meta_by_id = df_meta.set_index("image_id").to_dict(orient="index")

    # Step 2: Initialize Embedding Store & CLIP Model
    store = EmbeddingStore(cfg.embedding_dir)
    encoder = CLIPEncoder(model_name=cfg.clip_model_name, pretrained=cfg.clip_pretrained, device=args.device)
    store.set_model_metadata(cfg.clip_model_name, cfg.clip_pretrained, encoder.embedding_dim)

    completed_ids = store.get_completed_image_ids()
    logger.info("Embedding store has %d already embedded images.", len(completed_ids))

    temp_tar_dir = cfg.data_root / "temp_tars" / split
    temp_tar_dir.mkdir(parents=True, exist_ok=True)
    split_img_dir = cfg.image_dir / split
    split_img_dir.mkdir(parents=True, exist_ok=True)

    buffer_embeddings: List[np.ndarray] = []
    buffer_meta_rows: List[dict] = []
    current_shard_idx = store.get_next_shard_index()

    def flush_shard():
        nonlocal current_shard_idx, buffer_embeddings, buffer_meta_rows
        if not buffer_embeddings:
            return
        emb_arr = np.vstack(buffer_embeddings).astype(np.float32)
        meta_df = pd.DataFrame(buffer_meta_rows)
        store.write_shard(current_shard_idx, emb_arr, meta_df)
        logger.info("Flushed shard %05d: %d vectors (Total store vectors: %d)", current_shard_idx, len(emb_arr), store._manifest.get("total_vectors", 0))
        current_shard_idx += 1
        buffer_embeddings.clear()
        buffer_meta_rows.clear()

    # Step 3: Loop through chunks with asynchronous prefetching
    total_chunks = end_chunk - args.start_chunk + 1
    logger.info("Starting processing loop for %s chunks %d to %d (%d total chunks)...", split, args.start_chunk, end_chunk, total_chunks)

    prefetch_executor = ThreadPoolExecutor(max_workers=1)
    next_download_future: Optional[Future] = None

    for chunk_id in range(args.start_chunk, end_chunk + 1):
        tar_path = temp_tar_dir / f"images_{chunk_id:03d}.tar"
        logger.info("=== Processing %s Chunk [%d/%d] ===", split, chunk_id, end_chunk)

        # 1. Wait for current chunk download if prefetched, else download now
        if next_download_future is not None:
            download_ok = next_download_future.result()
            next_download_future = None
        else:
            download_ok = download_chunk(split, chunk_id, tar_path)

        if not download_ok or not tar_path.exists():
            logger.error("Download failed or archive missing for %s chunk %d, skipping.", split, chunk_id)
            continue

        # Trigger prefetch of the next chunk in background while extracting & embedding current chunk
        next_chunk_id = chunk_id + 1
        if next_chunk_id <= end_chunk:
            next_tar_path = temp_tar_dir / f"images_{next_chunk_id:03d}.tar"
            next_download_future = prefetch_executor.submit(download_chunk, split, next_chunk_id, next_tar_path)

        extract_tar(tar_path, split_img_dir)

        # 2. Find newly extracted images on disk
        chunk_files = []
        for root, _, files in os.walk(split_img_dir):
            for f in files:
                if f.endswith(".jpg"):
                    img_id = f[:-4]
                    if img_id not in completed_ids:
                        chunk_files.append(os.path.join(root, f))

        logger.info("Found %d pending images on disk for embedding in %s chunk %d.", len(chunk_files), split, chunk_id)

        # 3. Batch GPU Embedding with multi-worker DataLoader
        for i in range(0, len(chunk_files), args.batch_size * 4):
            batch_paths = chunk_files[i : i + args.batch_size * 4]
            embeddings, valid_paths, failed = encoder.encode_image_paths(
                batch_paths,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )

            if len(embeddings) > 0:
                buffer_embeddings.append(embeddings)
                for vp in valid_paths:
                    vid = Path(vp).stem
                    completed_ids.add(vid)
                    row_data = meta_by_id.get(vid, {})
                    buffer_meta_rows.append({
                        "image_id": vid,
                        "split": split,
                        "landmark_id": str(row_data.get("landmark_id", "")),
                        "landmark_name": str(row_data.get("landmark_name", "")),
                        "is_resolved": bool(row_data.get("is_resolved", False)),
                        "category_url": row_data.get("category_url", ""),
                        "hierarchical_label": row_data.get("hierarchical_label", ""),
                        "full_path": vp,
                        "relative_path": str(row_data.get("relative_path", "")),
                    })

            # Check if buffer reached shard_size
            curr_len = sum(len(b) for b in buffer_embeddings)
            if curr_len >= args.shard_size:
                flush_shard()

            if args.cleanup_images_after_embed:
                for vp in valid_paths:
                    try:
                        os.unlink(vp)
                    except OSError:
                        pass

        # Clean up chunk directory after chunk completes to keep disk lean
        if args.cleanup_images_after_embed:
            for root, dirs, files in os.walk(split_img_dir, topdown=False):
                for f in files:
                    if f.endswith(".jpg"):
                        try:
                            os.unlink(os.path.join(root, f))
                        except OSError:
                            pass
                for d in dirs:
                    try:
                        os.rmdir(os.path.join(root, d))
                    except OSError:
                        pass

        logger.info("Completed %s chunk %d. Embedded so far: %d", split, chunk_id, len(completed_ids))

    # Final flush
    flush_shard()
    logger.info("=== STREAMING PIPELINE FINISHED ===")
    logger.info("Total embedded vectors in store: %d across %d shards.", store._manifest.get("total_vectors", 0), len(store._manifest.get("shards", [])))


if __name__ == "__main__":
    main()
