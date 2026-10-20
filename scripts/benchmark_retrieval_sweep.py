#!/usr/bin/env python3
"""Benchmark ANN Recall@K vs Exact Search and Latency Sweep for IVF-PQ and HNSW-PQ."""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.index.ivfpq import IVFPQIndex
from src.index.hnsw_pq import HNSWPQIndex
from src.storage.embedding_store import EmbeddingStore
from src.utils.logging import get_logger

logger = get_logger("benchmark_sweep")


def compute_exact_ground_truth(
    queries: np.ndarray,
    store: EmbeddingStore,
    df_mapping: pd.DataFrame,
    top_k: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute true exact top-K inner-product nearest neighbors across all 5.05M vectors."""
    n_queries = len(queries)
    logger.info("Computing exact ground truth across all 5.05M vectors for %d queries...", n_queries)

    # Global min-heap / running top-k tracker
    # We maintain the top_k highest scores and corresponding faiss_ids for each query
    best_scores = np.full((n_queries, top_k), -np.inf, dtype=np.float32)
    best_ids = np.full((n_queries, top_k), -1, dtype=np.int64)

    # Shard metadata mapping lookup
    shard_fids = {}
    for shard_name, grp in df_mapping.groupby("shard_name", sort=False):
        # Sort by shard_row_idx to ensure direct alignment with mmap
        sorted_grp = grp.sort_values("shard_row_idx")
        shard_fids[str(shard_name)] = sorted_grp["faiss_id"].values.astype(np.int64)

    t0 = time.perf_counter()
    shard_count = 0

    for shard_info, emb_mmap, _ in store.iter_shards(mmap=True):
        shard_str = shard_info["shard_name"] if isinstance(shard_info, dict) else str(shard_info)
        if shard_str not in shard_fids:
            continue

        fids = shard_fids[shard_str]
        n_shard = len(fids)
        if n_shard == 0 or len(emb_mmap) < n_shard:
            continue

        # Compute dot products: (n_queries, n_shard)
        shard_scores = queries @ emb_mmap[:n_shard].T

        # Update running top_k per query
        for q_idx in range(n_queries):
            cur_sc = shard_scores[q_idx]
            # Fast filter: only examine candidates that beat the current k-th score
            cutoff = best_scores[q_idx, -1]
            candidate_mask = cur_sc > cutoff
            if not np.any(candidate_mask):
                continue

            cand_sc = cur_sc[candidate_mask]
            cand_id = fids[candidate_mask]

            combined_sc = np.concatenate([best_scores[q_idx], cand_sc])
            combined_id = np.concatenate([best_ids[q_idx], cand_id])

            top_idx = np.argpartition(-combined_sc, top_k)[:top_k]
            top_idx = top_idx[np.argsort(-combined_sc[top_idx])]

            best_scores[q_idx] = combined_sc[top_idx]
            best_ids[q_idx] = combined_id[top_idx]

        shard_count += 1
        if shard_count % 100 == 0:
            logger.info("Processed %d / 494 shards...", shard_count)

    elapsed = time.perf_counter() - t0
    logger.info("Exact ground truth computed in %.2f seconds.", elapsed)
    return best_scores, best_ids


def compute_pool_recall(approx_pool: np.ndarray, exact_ground_truth: np.ndarray, k: int) -> float:
    """Compute recall: fraction of true exact top-k that appear in the candidate pool."""
    n_queries = len(exact_ground_truth)
    total_recall = 0.0

    for i in range(n_queries):
        true_k = set(int(x) for x in exact_ground_truth[i, :k] if x >= 0)
        if not true_k:
            continue
        pool_set = set(int(x) for x in approx_pool[i] if x >= 0)
        hits = len(true_k.intersection(pool_set))
        total_recall += hits / len(true_k)

    return total_recall / max(1, n_queries)


def run_benchmark():
    mapping_path = Path("outputs/index/id_mapping.parquet")
    ivf_index_path = Path("outputs/index/ivfpq_populated.index")
    hnsw_index_path = Path("outputs/index/hnswpq_populated.index")
    emb_dir = Path("outputs/embeddings")

    logger.info("Loading ID mapping table...")
    df_mapping = pd.read_parquet(mapping_path)

    # Sample 50 evaluation queries from shard 0 with fixed seed
    logger.info("Sampling 50 evaluation queries...")
    first_shard = np.load(emb_dir / "shard_00000_emb.npy", mmap_mode="r")
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(first_shard), size=50, replace=False)
    queries = np.array(first_shard[sample_indices], dtype=np.float32, copy=True)
    # Ensure L2 normalization
    q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries = queries / np.maximum(q_norms, 1e-12)

    store = EmbeddingStore(emb_dir)
    exact_scores, exact_ids = compute_exact_ground_truth(queries, store, df_mapping, top_k=100)

    # -------------------------------------------------------------
    # 1. IVF-PQ Sweep
    # -------------------------------------------------------------
    logger.info("Loading IVF-PQ index...")
    ivfpq = IVFPQIndex()
    ivfpq.load(ivf_index_path)

    nprobe_list = [8, 16, 32, 64, 128, 256]
    candidate_k_list = [100, 500, 1000]

    ivf_results = []
    print("\nRunning IVF-PQ Sweep...")
    for nprobe in nprobe_list:
        ivfpq.set_nprobe(nprobe)
        for cand_k in candidate_k_list:
            latencies = []
            pool_ids = []
            for q in queries:
                t0 = time.perf_counter()
                sc, ids = ivfpq.search(q.reshape(1, -1), top_k=cand_k)
                lat = (time.perf_counter() - t0) * 1000.0
                latencies.append(lat)
                pool_ids.append(ids[0])

            pool_ids_arr = np.array(pool_ids)
            r10 = compute_pool_recall(pool_ids_arr, exact_ids, k=10)
            r100 = compute_pool_recall(pool_ids_arr, exact_ids, k=100)
            p50 = float(np.percentile(latencies, 50))
            p95 = float(np.percentile(latencies, 95))

            ivf_results.append({
                "nprobe": nprobe,
                "candidate_k": cand_k,
                "recall@10": r10,
                "recall@100": r100,
                "p50_ms": p50,
                "p95_ms": p95,
            })
            print(f"IVF-PQ | nprobe={nprobe:3d} cand_k={cand_k:4d} | R@10={r10:.4f} R@100={r100:.4f} | p50={p50:5.2f}ms p95={p95:5.2f}ms")

    # -------------------------------------------------------------
    # 2. HNSW-PQ Sweep
    # -------------------------------------------------------------
    logger.info("Loading HNSW-PQ index...")
    hnsw = HNSWPQIndex()
    hnsw.load(hnsw_index_path)

    ef_search_list = [64, 128, 256, 512]
    hnsw_results = []
    print("\nRunning HNSW-PQ Sweep...")
    for ef in ef_search_list:
        hnsw.set_ef_search(ef)
        for cand_k in candidate_k_list:
            latencies = []
            pool_ids = []
            for q in queries:
                t0 = time.perf_counter()
                sc, ids = hnsw.search(q.reshape(1, -1), top_k=cand_k)
                lat = (time.perf_counter() - t0) * 1000.0
                latencies.append(lat)
                pool_ids.append(ids[0])

            pool_ids_arr = np.array(pool_ids)
            r10 = compute_pool_recall(pool_ids_arr, exact_ids, k=10)
            r100 = compute_pool_recall(pool_ids_arr, exact_ids, k=100)
            p50 = float(np.percentile(latencies, 50))
            p95 = float(np.percentile(latencies, 95))

            hnsw_results.append({
                "ef_search": ef,
                "candidate_k": cand_k,
                "recall@10": r10,
                "recall@100": r100,
                "p50_ms": p50,
                "p95_ms": p95,
            })
            print(f"HNSW-PQ | ef={ef:3d} cand_k={cand_k:4d} | R@10={r10:.4f} R@100={r100:.4f} | p50={p50:5.2f}ms p95={p95:5.2f}ms")

    # Output formatted report
    report = {
        "ivf_results": ivf_results,
        "hnsw_results": hnsw_results,
    }
    with open("outputs/evaluation/retrieval_sweep_benchmark.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\nBenchmark completed. Results written to outputs/evaluation/retrieval_sweep_benchmark.json")


if __name__ == "__main__":
    run_benchmark()
