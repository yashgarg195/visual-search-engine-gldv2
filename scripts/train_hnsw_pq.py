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

    # 1. Sample training vectors
    actual_sample_size = min(args.sample_size, total_available)
    logger.info("Sampling %d representative vectors for HNSW-PQ codebook training...", actual_sample_size)
    t0 = time.time()
    train_vectors = store.sample_training_vectors(actual_sample_size, seed=cfg.random_seed)
    logger.info("Sampling completed in %.2f seconds.", time.time() - t0)

    # 2. Initialize HNSWPQIndex
    d = train_vectors.shape[1]
    validate_hnsw_pq_parameters(d=d, m=args.m, M=args.M, num_training_vectors=len(train_vectors))

    index = HNSWPQIndex(
        d=d,
        m=args.m,
        M=args.M,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )

    # 3. Train codebooks
    logger.info("Training HNSW-PQ index (d=%d, m=%d, M=%d, efConstruction=%d)...", d, args.m, args.M, args.ef_construction)
    index.train(train_vectors)

    # 4. Populate index with vectors and preserve ID mappings
    target_vectors = total_available if args.max_vectors is None else min(args.max_vectors, total_available)
    logger.info("Populating HNSW graph with up to %d vectors from persistent shards...", target_vectors)

    out_index_path = cfg.index_dir / args.output_name
    added_count = 0
    t_add_start = time.time()

    for shard in store.manifest.get("shards", []):
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

        index.add(np.ascontiguousarray(arr, dtype=np.float32), shard_ids)
        added_count += len(arr)

        if added_count % 50000 == 0 or added_count == target_vectors:
            rate = added_count / max(1e-3, time.time() - t_add_start)
            logger.info("Added %d / %d vectors (%.1f vecs/sec)...", added_count, target_vectors, rate)

    # 5. Persist populated HNSW-PQ index
    index.save(out_index_path)
    logger.info(
        "Successfully built and persisted HNSW-PQ index with %d vectors to %s!",
        index.total_vectors,
        out_index_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
