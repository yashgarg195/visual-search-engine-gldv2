#!/usr/bin/env python3
"""Train and populate a FAISS HNSW-PQ (Graph + Product Quantization) index from persistent embedding shards.

Supports training on a representative random sample of vectors and populating
the Hierarchical Navigable Small World graph with companion metadata mappings.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.index.hnsw_pq import HNSWPQIndex, validate_hnsw_pq_parameters
from src.storage.embedding_store import EmbeddingStore
from src.utils.io import atomic_save_json, atomic_save_parquet, load_json
from src.utils.logging import get_logger

logger = get_logger("train_hnsw_pq")


def main():
    parser = argparse.ArgumentParser(
        description="Train and populate a FAISS HNSW-PQ index from embedding shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", type=str, default=".env", help="Path to .env configuration file")
    parser.add_argument("--sample-size", type=int, default=50000, help="Number of vectors used for PQ codebook training")
    parser.add_argument("--max-vectors", type=int, default=None, help="Maximum vectors to add (None for all available)")
    parser.add_argument("--M", type=int, default=32, help="Number of bidirectional graph links per vector")
    parser.add_argument("--m", type=int, default=64, help="Number of PQ subquantizers (must divide dimension)")
    parser.add_argument("--ef-construction", type=int, default=64, help="HNSW graph construction exploration depth")
    parser.add_argument("--ef-search", type=int, default=64, help="Default runtime search exploration depth")
    parser.add_argument("--num-threads", type=int, default=4, help="Number of CPU cores/threads for parallel graph insertion")
    parser.add_argument("--checkpoint-interval", type=int, default=50, help="Checkpoint intermediate index to disk every N shards")
    parser.add_argument("--reuse-trained", action="store_true", default=False, help="Reuse already-trained index codebooks and reset graph")
    parser.add_argument("--output-name", type=str, default="hnswpq_populated.index", help="Output filename inside index_dir")
    args = parser.parse_args()

    cfg = get_config(args.env_file)
    cfg.ensure_directories()

    store = EmbeddingStore(cfg.embedding_dir)
    total_available = store.total_vectors
    if total_available == 0:
        logger.error("No embedding shards found in %s. Please generate embeddings first.", cfg.embedding_dir)
        return 1

    logger.info("Found %d total vectors across %d shards.", total_available, store.total_shards)
    out_index_path = cfg.index_dir / args.output_name

    # 1. Initialize or load trained index
    index = None
    if args.reuse_trained and out_index_path.is_file():
        try:
            logger.info("Loading existing trained index from %s to reuse codebooks...", out_index_path)
            candidate_index = HNSWPQIndex(
                d=512,
                m=args.m,
                M=args.M,
                ef_construction=args.ef_construction,
                ef_search=args.ef_search,
            )
            candidate_index.load(out_index_path)
            if candidate_index.is_trained:
                logger.info("Existing index is trained! Resetting vectors for fresh 5M population...")
                candidate_index.reset()
                index = candidate_index
        except Exception as e:
            logger.warning("Could not reuse existing index: %s. Training from scratch.", e)

    if index is None:
        # Sample training vectors
        actual_sample_size = min(args.sample_size, total_available)
        logger.info("Sampling %d representative vectors for HNSW-PQ codebook training...", actual_sample_size)
        t0 = time.time()
        train_vectors = store.sample_training_vectors(actual_sample_size, seed=cfg.random_seed)
        logger.info("Sampling completed in %.2f seconds.", time.time() - t0)

        d = train_vectors.shape[1]
        validate_hnsw_pq_parameters(d=d, m=args.m, M=args.M, num_training_vectors=len(train_vectors))

        index = HNSWPQIndex(
            d=d,
            m=args.m,
            M=args.M,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
        )

        logger.info("Training HNSW-PQ index (d=%d, m=%d, M=%d, efConstruction=%d)...", d, args.m, args.M, args.ef_construction)
        index.train(train_vectors)

    # 2. Populate index with vectors and preserve ID mappings
    target_vectors = total_available if args.max_vectors is None else min(args.max_vectors, total_available)
    logger.info(
        "Populating HNSW graph with %d vectors using %d CPU threads (checkpointing every %d shards)...",
        target_vectors,
        args.num_threads,
        args.checkpoint_interval,
    )

    out_index_path = cfg.index_dir / args.output_name
    added_count = 0
    t_add_start = time.time()

    shards = store.manifest.get("shards", [])
    for shard_idx, shard in enumerate(shards):
        if added_count >= target_vectors:
            break

        shard_name = shard["shard_name"]
        emb_path = cfg.embedding_dir / shard["emb_file"]
        meta_path = cfg.embedding_dir / shard["meta_file"]

        if not emb_path.is_file() or not meta_path.is_file():
            continue

        arr = np.load(emb_path, mmap_mode="r")
        df_meta = pd.read_parquet(meta_path)

        # Slice if near target
        remaining = target_vectors - added_count
        if len(arr) > remaining:
            arr = arr[:remaining]
            df_meta = df_meta.iloc[:remaining]

        # Generate integer IDs offset from current added count
        shard_ids = np.arange(added_count, added_count + len(arr), dtype=np.int64)

        index.add(np.ascontiguousarray(arr, dtype=np.float32), shard_ids, num_threads=args.num_threads)
        added_count += len(arr)

        if added_count % 50000 == 0 or added_count == target_vectors:
            rate = added_count / max(1e-3, time.time() - t_add_start)
            eta_sec = (target_vectors - added_count) / max(1e-3, rate)
            logger.info(
                "Progress: %d / %d vectors (%.1f vecs/sec, ETA: %.1f min)...",
                added_count,
                target_vectors,
                rate,
                eta_sec / 60.0,
            )

        # Periodic checkpoint
        if (shard_idx + 1) % args.checkpoint_interval == 0:
            logger.info("Saving checkpoint at %d vectors to %s...", index.total_vectors, out_index_path)
            index.save(out_index_path)

    # 5. Persist fully populated HNSW-PQ index
    index.save(out_index_path)
    t_total = time.time() - t_add_start
    logger.info(
        "Successfully built and persisted HNSW-PQ index with %d vectors in %.1f min (%.1f vecs/sec) to %s!",
        index.total_vectors,
        t_total / 60.0,
        index.total_vectors / max(1e-3, t_total),
        out_index_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
