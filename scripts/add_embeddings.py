#!/usr/bin/env python3
"""Batch Ingestion of Embedding Shards into FAISS IVF-PQ Index with Explicit ID Tracking.

Streams shards one-by-one to avoid high memory pressure, assigns explicit int64 IDs,
maintains a bi-directional mapping (faiss_id <-> image_id <-> landmark_id),
and supports atomic checkpointing to prevent duplicate additions on resumption.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Set
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.index.ivfpq import IVFPQIndex
from src.storage.embedding_store import EmbeddingStore
from src.utils.io import atomic_save_json, atomic_save_parquet, load_json
from src.utils.logging import get_logger

logger = get_logger("add_embeddings_cli")


def main():
    parser = argparse.ArgumentParser(
        description="Add embedding shards to trained FAISS IVF-PQ index with explicit ID tracking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", type=str, default=".env", help="Path to .env configuration file")
    parser.add_argument(
        "--trained-index",
        type=str,
        default=None,
        help="Path to trained base index (default: outputs/index/ivfpq_trained.index)",
    )
    parser.add_argument(
        "--output-index",
        type=str,
        default=None,
        help="Path to save populated index (default: outputs/index/ivfpq_populated.index)",
    )
    parser.add_argument("--save-every-shard", action="store_true", default=True, help="Checkpoint index after every shard")
    parser.add_argument("--force-rebuild", action="store_true", help="Clear existing populated index and rebuild from scratch")
    args = parser.parse_args()

    cfg = get_config(args.env_file)
    cfg.ensure_directories()

    trained_index_path = (
        Path(args.trained_index) if args.trained_index else cfg.index_dir / "ivfpq_trained.index"
    )
    output_index_path = (
        Path(args.output_index) if args.output_index else cfg.index_dir / "ivfpq_populated.index"
    )
    index_manifest_path = cfg.index_dir / "index_manifest.json"
    id_mapping_path = cfg.index_dir / "id_mapping.parquet"

    store = EmbeddingStore(cfg.embedding_dir)
    total_store_vectors = store._manifest.get("total_vectors", 0)

    if total_store_vectors == 0:
        logger.error("No embedding shards found in %s", cfg.embedding_dir)
        return 1

    # Load manifest or create new
    manifest: Dict[str, Any] = {
        "shards_added": [],
        "total_added": 0,
        "last_faiss_id": 0,
    }
    existing_mappings: List[pd.DataFrame] = []

    if output_index_path.is_file() and not args.force_rebuild:
        if index_manifest_path.is_file():
            try:
                manifest = load_json(index_manifest_path)
            except Exception as e:
                logger.warning("Could not read index manifest: %s", e)

        if id_mapping_path.is_file():
            try:
                existing_mappings.append(pd.read_parquet(id_mapping_path))
            except Exception as e:
                logger.warning("Could not read existing id_mapping: %s", e)

        # Load populated index
        index = IVFPQIndex()
        index.load(output_index_path)
        logger.info("Resuming from populated index (%d vectors already indexed).", index.total_vectors)
    else:
        if not trained_index_path.is_file():
            logger.error("Trained index not found at %s. Run scripts/train_ivfpq.py first.", trained_index_path)
            return 1
        index = IVFPQIndex()
        index.load(trained_index_path)
        logger.info("Loaded fresh trained index from %s", trained_index_path)

    shards_already_added: Set[str] = set(manifest.get("shards_added", []))
    all_shards = store._manifest.get("shards", [])
    pending_shards = [s for s in all_shards if s["shard_name"] not in shards_already_added]

    logger.info(
        "Total shards in store: %d | Already added: %d | Pending addition: %d",
        len(all_shards),
        len(shards_already_added),
        len(pending_shards),
    )

    if not pending_shards:
        logger.info("All embedding shards are already indexed. No work to do.")
        return 0

    current_faiss_id = int(manifest.get("last_faiss_id", 0))
    new_mapping_dfs: List[pd.DataFrame] = []

    for shard_info in tqdm(pending_shards, desc="Adding Shards to IVF-PQ"):
        shard_name = shard_info["shard_name"]
        emb_file = cfg.embedding_dir / shard_info["emb_file"]
        meta_file = cfg.embedding_dir / shard_info["meta_file"]

        if not emb_file.is_file() or not meta_file.is_file():
            logger.error("Shard files missing for %s, skipping.", shard_name)
            continue

        # Load vectors and companion metadata
        vectors = np.load(emb_file, mmap_mode="r")
        meta_df = pd.read_parquet(meta_file)

        if len(vectors) != len(meta_df):
            logger.error("Mismatch in shard %s: %d vectors vs %d rows", shard_name, len(vectors), len(meta_df))
            continue

        n_vecs = len(vectors)
        faiss_ids = np.arange(current_faiss_id, current_faiss_id + n_vecs, dtype=np.int64)

        # Add to FAISS index with explicit IDs
        index.add(vectors=np.array(vectors, copy=False, dtype=np.float32), ids=faiss_ids)

        # Build mapping records
        mapping_df = pd.DataFrame({
            "faiss_id": faiss_ids,
            "image_id": meta_df["image_id"].astype(str),
            "landmark_id": meta_df["landmark_id"].astype(str),
            "split": meta_df.get("split", "train").astype(str),
            "shard_name": shard_name,
            "shard_row_idx": np.arange(n_vecs, dtype=np.int64),
        })
        new_mapping_dfs.append(mapping_df)

        current_faiss_id += n_vecs
        manifest["shards_added"].append(shard_name)
        manifest["total_added"] = index.total_vectors
        manifest["last_faiss_id"] = current_faiss_id

        if args.save_every_shard:
            index.save(output_index_path)
            atomic_save_json(manifest, index_manifest_path)

            # Consolidate mapping dataframe
            combined_mappings = pd.concat(existing_mappings + new_mapping_dfs, ignore_index=True)
            atomic_save_parquet(combined_mappings, id_mapping_path)

    # Final save
    index.save(output_index_path)
    atomic_save_json(manifest, index_manifest_path)
    combined_mappings = pd.concat(existing_mappings + new_mapping_dfs, ignore_index=True)
    atomic_save_parquet(combined_mappings, id_mapping_path)

    print("\n" + "=" * 70)
    print(" EMBEDDING INGESTION COMPLETE")
    print("=" * 70)
    print(f"  Total vectors in index : {index.total_vectors}")
    print(f"  Total shards added     : {len(manifest['shards_added'])}")
    print(f"  Saved populated index  : {output_index_path}")
    print(f"  Saved ID mapping table : {id_mapping_path} ({len(combined_mappings)} records)\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
