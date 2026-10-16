#!/usr/bin/env python3
"""Train FAISS IVF-PQ Index on Stored Embedding Shards.

Samples a representative subset of normalized CLIP embeddings, validates
IVF-PQ parameters (dimension divisibility, centroid counts, sample size),
trains the coarse quantizer and product quantizers, and records training metrics.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.index.ivfpq import IVFPQIndex
from src.storage.embedding_store import EmbeddingStore
from src.utils.logging import get_logger

logger = get_logger("train_ivfpq_cli")


def main():
    parser = argparse.ArgumentParser(
        description="Train FAISS IVF-PQ index on stored CLIP embeddings with strict parameter validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", type=str, default=".env", help="Path to .env configuration file")
    parser.add_argument("--nlist", type=int, default=None, help="Number of IVF cluster centroids")
    parser.add_argument("--nprobe", type=int, default=None, help="Default nprobe for search queries")
    parser.add_argument("--m", type=int, default=None, help="Number of PQ subquantizers (e.g. 64 for 512-dim)")
    parser.add_argument("--nbits", type=int, default=None, help="Bits per subquantizer code (default 8)")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of vectors to sample for training")
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument("--output-index", type=str, default=None, help="Custom path for trained index output")
    parser.add_argument("--force", action="store_true", help="Force retraining if trained index already exists")
    args = parser.parse_args()

    cfg = get_config(args.env_file)
    cfg.ensure_directories()

    nlist = args.nlist or cfg.faiss_nlist
    nprobe = args.nprobe or cfg.faiss_nprobe
    m = args.m or cfg.faiss_m
    nbits = args.nbits or cfg.faiss_nbits
    sample_size = args.sample_size or cfg.faiss_train_sample_size
    seed = args.random_seed or cfg.random_seed

    out_index_path = Path(args.output_index) if args.output_index else cfg.index_dir / "ivfpq_trained.index"

    if out_index_path.is_file() and not args.force:
        logger.info("Found existing trained index at %s. Use --force to retrain.", out_index_path)
        return 0

    store = EmbeddingStore(cfg.embedding_dir)
    total_vectors = store._manifest.get("total_vectors", 0)
    if total_vectors == 0:
        logger.error(
            "No embeddings found in store at %s. Run 'scripts/embed_images.py' before training.",
            cfg.embedding_dir,
        )
        return 1

    dim = store._manifest.get("embedding_dim", cfg.embedding_dim)

    # Automatically adapt nlist for small test runs if requested nlist > total_vectors
    if total_vectors < nlist:
        suggested_nlist = max(1, 2 ** int(np_floor_log2 := 4))  # e.g. 16 or 32
        while suggested_nlist * 2 <= total_vectors and suggested_nlist < 64:
            suggested_nlist *= 2
        logger.warning(
            "Dataset size (%d vectors) is smaller than requested nlist (%d). "
            "Automatically adjusting nlist to %d for valid clustering.",
            total_vectors,
            nlist,
            suggested_nlist,
        )
        nlist = suggested_nlist

    # Sample representative vectors
    try:
        training_vectors = store.sample_training_vectors(max_samples=sample_size, seed=seed)
    except Exception as e:
        logger.error("Failed to sample training vectors: %s", e)
        return 1

    # Initialize and train IVF-PQ
    try:
        index = IVFPQIndex(d=dim, nlist=nlist, m=m, nbits=nbits, nprobe=nprobe)
        index.train(training_vectors)
        index.save(out_index_path)
    except Exception as e:
        logger.error("IVF-PQ training failed: %s", e)
        return 1

    print("\n" + "=" * 70)
    print(" IVF-PQ INDEX TRAINING COMPLETED")
    print("=" * 70)
    print(f"  Dimension (d)       : {dim}")
    print(f"  Centroids (nlist)   : {nlist}")
    print(f"  Subquantizers (m)   : {m} (Bytes per vector = {m})")
    print(f"  Bits per code       : {nbits} (2^{nbits} = {2**nbits} centroids per subquantizer)")
    print(f"  Training sample     : {len(training_vectors)} vectors")
    print(f"  Index metric        : METRIC_INNER_PRODUCT (Cosine similarity for L2-normalized vectors)")
    print(f"  Saved index path    : {out_index_path}")
    print(f"  Artifact size       : {out_index_path.stat().st_size / 1024:.2f} KB\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
