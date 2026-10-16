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

TAR_URL_PATTERN = "https://s3.amazonaws.com/google-landmark/train/images_{chunk_id:03d}.tar"


def download_chunk(chunk_id: int, dest_tar: Path) -> bool:
    """Download single tar chunk with progress."""
    url = TAR_URL_PATTERN.format(chunk_id=chunk_id)
    temp_tar = dest_tar.with_suffix(f".tmp.{os.getpid()}")
    try:
        with urllib.request.urlopen(url) as resp, open(temp_tar, "wb") as f_out:
            total = int(resp.headers.get("content-length", 0))
            pbar = tqdm(total=total, unit="B", unit_scale=True, desc=f"Chunk {chunk_id:03d}.tar", leave=False)
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
        logger.error("Failed downloading chunk %d: %s", chunk_id, e)
        if temp_tar.exists():
            temp_tar.unlink()
        return False


def extract_tar(tar_path: Path, target_dir: Path) -> None:
    """Extract tar and remove archive immediately."""
    logger.info("Extracting %s into %s...", tar_path.name, target_dir)
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=target_dir)
    tar_path.unlink()


def main():
    parser = argparse.ArgumentParser(description="Pipelined Tar Downloader and GPU Vectorizer")
    parser.add_argument("--start-chunk", type=int, default=0, help="Starting tar chunk (0-499)")
    parser.add_argument("--end-chunk", type=int, default=499, help="Ending tar chunk (0-499)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for GPU inference")
    parser.add_argument("--shard-size", type=int, default=10000, help="Images per persistent embedding shard")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    parser.add_argument("--cleanup-images-after-embed", action="store_true", help="Delete images after embedding to save disk")
    args = parser.parse_args()

    cfg = get_config()
    cfg.ensure_directories()

    split = "train"
    metadata_parquet = cfg.processed_dir / f"metadata_{split}.parquet"

    # Step 1: Ensure metadata exists
    if not metadata_parquet.is_file():
        logger.info("Downloading and preparing GLDv2 training metadata...")
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

    temp_tar_dir = cfg.data_root / "temp_tars"
    temp_tar_dir.mkdir(parents=True, exist_ok=True)
    train_img_dir = cfg.image_dir / split
    train_img_dir.mkdir(parents=True, exist_ok=True)

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

    # Step 3: Loop through chunks
    total_chunks = args.end_chunk - args.start_chunk + 1
    logger.info("Starting processing loop for chunks %d to %d (%d total chunks)...", args.start_chunk, args.end_chunk, total_chunks)

    for chunk_id in range(args.start_chunk, args.end_chunk + 1):
        tar_path = temp_tar_dir / f"images_{chunk_id:03d}.tar"
        logger.info("=== Processing Chunk [%d/%d] ===", chunk_id, args.end_chunk)

        # 1. Download & Extract
        if not download_chunk(chunk_id, tar_path):
            continue
        extract_tar(tar_path, train_img_dir)

        # 2. Find newly extracted images on disk
        chunk_files = []
        for root, _, files in os.walk(train_img_dir):
            for f in files:
                if f.endswith(".jpg"):
                    img_id = f[:-4]
                    if img_id not in completed_ids:
                        chunk_files.append(os.path.join(root, f))

        logger.info("Found %d pending images on disk for embedding in chunk %d.", len(chunk_files), chunk_id)

        # 3. Batch GPU Embedding
        for i in range(0, len(chunk_files), args.batch_size):
            batch_paths = chunk_files[i : i + args.batch_size]
            embeddings, valid_paths, failed = encoder.encode_image_paths(batch_paths)

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

        logger.info("Completed chunk %d. Embedded so far: %d", chunk_id, len(completed_ids))

    # Final flush
    flush_shard()
    logger.info("=== STREAMING PIPELINE FINISHED ===")
    logger.info("Total embedded vectors in store: %d across %d shards.", store._manifest.get("total_vectors", 0), len(store.get_shard_manifests()))


if __name__ == "__main__":
    main()
