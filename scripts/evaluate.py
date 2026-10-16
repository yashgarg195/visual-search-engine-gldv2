#!/usr/bin/env python3
"""Comprehensive Retrieval & Recognition Evaluation Pipeline.

Measures:
1. Exact IndexFlatIP baseline comparison (ANN agreement: Recall@1, Recall@5, Recall@10).
2. Landmark Recognition Accuracy (Top-1, Top-5 accuracy vs. ground-truth landmark IDs).
3. Query Latency distribution (average, median, p95).
4. nprobe sweep to characterize the recall vs. latency trade-off curve.
5. Index size on disk and artifact metrics.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.index.flat import FlatIPIndex
from src.index.ivfpq import IVFPQIndex
from src.storage.embedding_store import EmbeddingStore
from src.utils.io import atomic_save_json
from src.utils.logging import get_logger

logger = get_logger("evaluate_cli")


def compute_ann_recall_at_k(
    approx_ids: np.ndarray,
    exact_ids: np.ndarray,
    k: int,
) -> float:
    """Compute Recall@K between approximate nearest neighbor results and exact ground truth.

    Recall@K = |Approx@K ∩ Exact@K| / K
    """
    total_recall = 0.0
    num_queries = len(exact_ids)
    if num_queries == 0 or k == 0:
        return 0.0

    for i in range(num_queries):
        true_set = set(exact_ids[i, :k])
        # Exclude invalid / padding negative IDs
        true_set.discard(-1)
        if not true_set:
            continue
        approx_set = set(approx_ids[i, :k])
        intersection = len(true_set.intersection(approx_set))
        total_recall += intersection / len(true_set)

    return total_recall / num_queries


def compute_landmark_accuracy_at_k(
    retrieved_faiss_ids: np.ndarray,
    query_landmark_ids: List[str],
    id_to_landmark_map: Dict[int, str],
    k: int,
) -> float:
    """Compute top-K landmark recognition accuracy.

    Returns fraction of queries where at least one of top-K retrieved images
    shares the exact ground-truth landmark ID.
    """
    correct = 0
    total = len(query_landmark_ids)
    if total == 0:
        return 0.0

    for i in range(total):
        target_lid = str(query_landmark_ids[i]).strip()
        matched = False
        for fid in retrieved_faiss_ids[i, :k]:
            if fid < 0:
                continue
            retrieved_lid = str(id_to_landmark_map.get(int(fid), "")).strip()
            if retrieved_lid and retrieved_lid == target_lid:
                matched = True
                break
        if matched:
            correct += 1

    return correct / total


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FAISS IVF-PQ retrieval quality, latency, and landmark recognition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", type=str, default=".env", help="Path to .env configuration file")
    parser.add_argument("--num-queries", type=int, default=100, help="Number of evaluation queries to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for query sampling")
    parser.add_argument(
        "--nprobe-sweep",
        nargs="+",
        type=int,
        default=[1, 4, 8, 16, 32, 64],
        help="List of nprobe values to evaluate",
    )
    parser.add_argument(
        "--exact-baseline-max-vectors",
        type=int,
        default=50000,
        help="Maximum vectors to include in exact IndexFlatIP baseline",
    )
    parser.add_argument("--output-report", type=str, default=None, help="Path for JSON evaluation report")
    args = parser.parse_args()

    cfg = get_config(args.env_file)
    cfg.ensure_directories()

    index_path = cfg.index_dir / "ivfpq_populated.index"
    id_mapping_path = cfg.index_dir / "id_mapping.parquet"

    if not index_path.is_file():
        logger.error("IVF-PQ index not found at %s. Please run add_embeddings.py first.", index_path)
        return 1

    if not id_mapping_path.is_file():
        logger.error("ID mapping table not found at %s.", id_mapping_path)
        return 1

    # 1. Load populated IVF-PQ index
    ivfpq = IVFPQIndex()
    ivfpq.load(index_path)

    # 2. Load ID mapping
    df_mapping = pd.read_parquet(id_mapping_path)
    id_to_landmark: Dict[int, str] = dict(zip(df_mapping["faiss_id"], df_mapping["landmark_id"].astype(str)))

    # 3. Load vectors for evaluation query set and exact baseline
    store = EmbeddingStore(cfg.embedding_dir)
    total_vectors = ivfpq.total_vectors
    if total_vectors < 5:
        logger.error("Index contains only %d vectors; need at least 5 for evaluation.", total_vectors)
        return 1

    # Load vectors from store
    all_vectors_list = []
    all_faiss_ids_list = []
    all_lids_list = []

    for _, v_chunk, m_chunk in store.iter_shards(mmap=True):
        all_vectors_list.append(np.array(v_chunk, copy=True, dtype=np.float32))
        all_lids_list.extend(m_chunk["landmark_id"].astype(str).tolist())

    all_vectors = np.vstack(all_vectors_list)
    dim = all_vectors.shape[1]

    # Sample evaluation queries
    rng = np.random.RandomState(args.seed)
    n_queries = min(args.num_queries, len(all_vectors))
    query_indices = rng.choice(len(all_vectors), size=n_queries, replace=False)

    query_vectors = all_vectors[query_indices]
    query_landmark_ids = [all_lids_list[idx] for idx in query_indices]

    logger.info("Sampled %d evaluation queries (dim=%d).", n_queries, dim)

    # 4. Build exact IndexFlatIP baseline
    baseline_limit = min(args.exact_baseline_max_vectors, len(all_vectors))
    logger.info("Building exact IndexFlatIP baseline over %d vectors...", baseline_limit)

    baseline_vectors = all_vectors[:baseline_limit]
    baseline_ids = df_mapping["faiss_id"].iloc[:baseline_limit].to_numpy(dtype=np.int64)

    flat_index = FlatIPIndex(d=dim)
    flat_index.add(baseline_vectors, baseline_ids)

    # Search exact baseline (k=10)
    flat_scores, flat_exact_ids = flat_index.search(query_vectors, top_k=10)

    # 5. Sweep nprobe on IVF-PQ
    sweep_results: List[Dict[str, Any]] = []

    print("\n" + "=" * 90)
    print(" EVALUATION: IVF-PQ RETRIEVAL QUALITY & LATENCY BENCHMARK")
    print("=" * 90)
    print(f" Total Indexed Vectors  : {total_vectors}")
    print(f" Exact Baseline Subset  : {baseline_limit} vectors")
    print(f" Evaluation Queries     : {n_queries} queries (Seed: {args.seed})")
    print(f" Index File Size        : {index_path.stat().st_size / (1024**2):.2f} MB")
    print("-" * 90)
    print(f"{'nprobe':<8} {'Recall@1':<12} {'Recall@5':<12} {'Recall@10':<12} {'Top-1 Acc':<12} {'Top-5 Acc':<12} {'Avg (ms)':<10} {'p95 (ms)'}")
    print("-" * 90)

    for nprobe in args.nprobe_sweep:
        ivfpq.set_nprobe(nprobe)

        latencies_ms = []
        retrieved_ids_list = []

        # Query one by one to capture accurate latency distribution
        for q_idx in range(n_queries):
            q_vec = query_vectors[q_idx : q_idx + 1]
            t0 = time.perf_counter()
            _, r_ids = ivfpq.search(q_vec, top_k=10)
            t1 = time.perf_counter()

            latencies_ms.append((t1 - t0) * 1000)
            retrieved_ids_list.append(r_ids[0])

        approx_ids = np.array(retrieved_ids_list)

        # 1. Compute ANN Recall against Exact IndexFlatIP (measuring ANN search agreement)
        r1 = compute_ann_recall_at_k(approx_ids, flat_exact_ids, k=1)
        r5 = compute_ann_recall_at_k(approx_ids, flat_exact_ids, k=5)
        r10 = compute_ann_recall_at_k(approx_ids, flat_exact_ids, k=10)

        # 2. Compute Landmark Recognition Accuracy vs Ground-Truth Labels
        acc1 = compute_landmark_accuracy_at_k(approx_ids, query_landmark_ids, id_to_landmark, k=1)
        acc5 = compute_landmark_accuracy_at_k(approx_ids, query_landmark_ids, id_to_landmark, k=5)

        avg_lat = float(np.mean(latencies_ms))
        med_lat = float(np.median(latencies_ms))
        p95_lat = float(np.percentile(latencies_ms, 95))

        print(
            f"{nprobe:<8} {r1:<12.4f} {r5:<12.4f} {r10:<12.4f} {acc1:<12.4f} {acc5:<12.4f} {avg_lat:<10.2f} {p95_lat:<.2f}"
        )

        sweep_results.append({
            "nprobe": nprobe,
            "ann_recall_at_1": round(r1, 4),
            "ann_recall_at_5": round(r5, 4),
            "ann_recall_at_10": round(r10, 4),
            "landmark_acc_at_1": round(acc1, 4),
            "landmark_acc_at_5": round(acc5, 4),
            "latency_ms": {
                "mean": round(avg_lat, 2),
                "median": round(med_lat, 2),
                "p95": round(p95_lat, 2),
            },
        })

    print("=" * 90)
    print(" METRICS CLARIFICATION:")
    print("  - ANN Recall@K      : Agreement with exact IndexFlatIP nearest neighbours.")
    print("  - Landmark Acc@K    : Ground-truth landmark ID classification accuracy.")
    print("=" * 90 + "\n")

    # Save evaluation report
    report_path = (
        Path(args.output_report) if args.output_report else cfg.eval_dir / "evaluation_report.json"
    )
    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_indexed_vectors": total_vectors,
        "exact_baseline_vectors": baseline_limit,
        "num_queries": n_queries,
        "index_size_mb": round(index_path.stat().st_size / (1024**2), 2),
        "sweep_results": sweep_results,
    }
    atomic_save_json(report_data, report_path)
    logger.info("Saved evaluation report to %s", report_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
