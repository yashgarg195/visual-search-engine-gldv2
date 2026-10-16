#!/usr/bin/env python3
"""Resumable Sharded Image Embedding Generation using OpenCLIP.

Processes verified GLDv2 images, computes L2-normalized embeddings in batches,
records corrupt/missing images, writes incremental atomic shards, and tracks
progress in a global manifest to guarantee safe resumption without duplication.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.models.clip_encoder import CLIPEncoder
from src.storage.embedding_store import EmbeddingStore
from src.utils.io import atomic_save_json
from src.utils.logging import get_logger

logger = get_logger("embed_images_cli")


def main():
    parser = argparse.ArgumentParser(
        description="Extract normalized CLIP embeddings in persistent shards with checkpoint resumption.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", type=str, default=".env", help="Path to .env configuration file")
    parser.add_argument("--split", type=str, default="train", choices=["train", "index"], help="Split to embed")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for model inference")
    parser.add_argument("--shard-size", type=int, default=None, help="Number of images per persistent shard")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('auto', 'cuda', 'mps', 'cpu')")
    parser.add_argument("--max-images", type=int, default=None, help="Max images to process in this run")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="OpenCLIP architecture (e.g. ViT-B-32)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Pretrained weights tag (e.g. openai, laion2b_s34b_b79k)",
    )
    args = parser.parse_args()

    cfg = get_config(args.env_file)
    cfg.ensure_directories()

    batch_size = args.batch_size or cfg.batch_size
    shard_size = args.shard_size or cfg.shard_size
    device = args.device or cfg.device
    model_name = args.model_name or cfg.clip_model_name
    pretrained = args.pretrained or cfg.clip_pretrained

    # 1. Load validated metadata
    parquet_path = cfg.processed_dir / f"metadata_{args.split}.parquet"
    if not parquet_path.is_file():
        logger.error(
            "Validated metadata parquet file not found at %s. "
            "Please run 'python scripts/prepare_metadata.py --splits %s' first.",
            parquet_path,
            args.split,
        )
        return 1

    df_meta = pd.read_parquet(parquet_path)
    # Filter to existing images only
    if "exists" in df_meta.columns:
        df_meta = df_meta[df_meta["exists"] == True].copy()

    if len(df_meta) == 0:
        logger.error("No valid image files found in %s to process.", parquet_path)
        return 1

    # 2. Initialize embedding store and inspect completed IDs
    store = EmbeddingStore(cfg.embedding_dir)
    completed_ids = store.get_completed_image_ids()
    logger.info("Found %d already embedded images in store.", len(completed_ids))

    # Filter out already embedded images
    df_to_embed = df_meta[~df_meta["image_id"].isin(completed_ids)].copy()
    if args.max_images:
        df_to_embed = df_to_embed.iloc[: args.max_images].copy()

    total_pending = len(df_to_embed)
    logger.info("Total images pending embedding for split '%s': %d", args.split, total_pending)

    if total_pending == 0:
        logger.info("All images are already embedded! No work needed.")
        return 0

    # 3. Initialize model
    encoder = CLIPEncoder(model_name=model_name, pretrained=pretrained, device=device)
    store.set_model_metadata(
        model_name=model_name,
        pretrained=pretrained,
        embedding_dim=encoder.embedding_dim,
    )

    # 4. Process images in batches and write shards
    shard_embeddings_buffer: List[np.ndarray] = []
    shard_meta_rows: List[dict] = []
    failed_images_log: List[Tuple[str, str]] = []

    current_shard_idx = store.get_next_shard_index()
    start_time = time.time()
    total_processed = 0

    pbar = tqdm(total=total_pending, desc=f"Embedding {args.split}")

    # Helper to flush buffer to a shard
    def flush_shard():
        nonlocal current_shard_idx, shard_embeddings_buffer, shard_meta_rows
        if not shard_embeddings_buffer:
            return
        emb_array = np.vstack(shard_embeddings_buffer).astype(np.float32)
        meta_df = pd.DataFrame(shard_meta_rows)

        store.write_shard(
            shard_idx=current_shard_idx,
            embeddings=emb_array,
            metadata_df=meta_df,
        )
        current_shard_idx += 1
        shard_embeddings_buffer.clear()
        shard_meta_rows.clear()

    # Iterate over records in chunks of batch_size
    records = df_to_embed.to_dict(orient="records")
    for i in range(0, len(records), batch_size):
        chunk = records[i : i + batch_size]
        paths = [r["full_path"] for r in chunk]

        embeddings, valid_paths, failed = encoder.encode_image_paths(paths)

        if failed:
            failed_images_log.extend(failed)
            for fail_p, err in failed:
                logger.warning("Corrupt image skipped: %s (%s)", fail_p, err)

        # Match successfully encoded images with their metadata
        valid_path_set = set(valid_paths)
        valid_chunk = [r for r in chunk if r["full_path"] in valid_path_set]

        if len(valid_chunk) > 0 and len(embeddings) > 0:
            shard_embeddings_buffer.append(embeddings)
            shard_meta_rows.extend(valid_chunk)
            total_processed += len(valid_chunk)

        pbar.update(len(chunk))

        # Check if shard buffer has reached shard_size
        current_buffer_len = sum(len(b) for b in shard_embeddings_buffer)
        if current_buffer_len >= shard_size:
            flush_shard()

    # Final flush for any remaining images
    flush_shard()
    pbar.close()

    elapsed = time.time() - start_time
    logger.info("Completed embedding run: %d images in %.2f seconds (%.1f img/s)", total_processed, elapsed, (total_processed / max(1e-5, elapsed)))

    # Save failed images report if any
    if failed_images_log:
        err_path = cfg.embedding_dir / "failed_images.json"
        atomic_save_json(failed_images_log, err_path)
        logger.warning("Recorded %d corrupt/failed images to %s", len(failed_images_log), err_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
