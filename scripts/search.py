#!/usr/bin/env python3
"""Visual Landmark Search with Pretrained CLIP, FAISS IVF-PQ, and Weighted Voting.

Accepts an image query (or text query), extracts normalized CLIP embeddings,
queries the IVF-PQ index, resolves landmark names, performs similarity-weighted voting
to identify the leading candidate, and prints or exports structured results.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.index.ivfpq import IVFPQIndex
from src.index.hnsw_pq import HNSWPQIndex
from src.index.rerank import TwoStageReranker
from src.index.voting import compute_similarity_weighted_votes
from src.metadata.lookup import LandmarkLookupTable
from src.models.clip_encoder import CLIPEncoder
from src.utils.logging import get_logger

logger = get_logger("search_cli")


def main():
    parser = argparse.ArgumentParser(
        description="Query the IVF-PQ or HNSW-PQ landmark search index with an image or text prompt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", type=str, default=".env", help="Path to .env configuration file")
    parser.add_argument("--query-image", type=str, default=None, help="Path to query image file")
    parser.add_argument("--query-text", type=str, default=None, help="Optional text prompt query")
    parser.add_argument("--top-k", type=int, default=10, help="Number of nearest neighbors to retrieve")
    parser.add_argument(
        "--index-type",
        type=str,
        choices=["ivfpq", "hnswpq", "ensemble"],
        default="ivfpq",
        help="Vector index architecture to query ('ivfpq', 'hnswpq', or 'ensemble')",
    )
    parser.add_argument("--nprobe", type=int, default=64, help="Number of IVF Voronoi cells to probe at query time")
    parser.add_argument("--ef-search", type=int, default=64, help="Exploration depth for HNSW graph search")
    parser.add_argument("--rerank", action="store_true", help="Perform two-stage exact reranking on candidates")
    parser.add_argument("--rerank-candidates", type=int, default=50, help="Candidate pool size for two-stage rerank")
    parser.add_argument("--alpha-qe", action="store_true", help="Perform fast α-Query Expansion vector reranking")
    parser.add_argument("--geometric-rerank", action="store_true", help="Perform Stage-2 local feature geometric verification")
    parser.add_argument(
        "--matcher",
        type=str,
        choices=["sift", "lightglue"],
        default="sift",
        help="Feature matching algorithm for Stage-2 ('sift' or 'lightglue')",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to use ('auto', 'cuda', 'mps', 'cpu')")
    parser.add_argument("--pretrained", type=str, default=None, help="Pretrained weights tag (e.g. openai, none)")
    parser.add_argument("--exclude-image-id", type=str, default=None, help="Exclude specific image ID from results")
    parser.add_argument("--output-csv", type=str, default=None, help="Save search results to CSV file")
    parser.add_argument("--output-json", type=str, default=None, help="Save search results to JSON file")

    args = parser.parse_args()

    cfg = get_config(args.env_file)

    if not args.query_image and not args.query_text:
        logger.error("Please provide either --query-image <path> or --query-text '<prompt>'")
        return 1

    # 1. Load FAISS index based on index-type
    index_ivf = None
    index_hnsw = None
    index = None

    if args.index_type in ("hnswpq", "ensemble"):
        hnsw_path = cfg.index_dir / "hnswpq_populated.index"
        if not hnsw_path.is_file():
            logger.error("HNSW-PQ index not found at %s. Please train index first.", hnsw_path)
            return 1
        index_hnsw = HNSWPQIndex()
        index_hnsw.load(hnsw_path)
        index_hnsw.set_ef_search(args.ef_search)
        logger.info("Loaded HNSW-PQ index (%d vectors, efSearch=%d)", index_hnsw.total_vectors, args.ef_search)
        index = index_hnsw

    if args.index_type in ("ivfpq", "ensemble"):
        ivf_path = cfg.index_dir / "ivfpq_populated.index"
        if not ivf_path.is_file():
            ivf_path = cfg.index_dir / "ivfpq_trained.index"
            if not ivf_path.is_file():
                logger.error("FAISS IVF-PQ index not found at %s.", ivf_path)
                return 1
        index_ivf = IVFPQIndex()
        index_ivf.load(ivf_path)
        index_ivf.set_nprobe(args.nprobe)
        logger.info("Loaded IVF-PQ index (%d vectors, nprobe=%d)", index_ivf.total_vectors, args.nprobe)
        if index is None:
            index = index_ivf

    total_vecs = index.total_vectors if index else 0
    if total_vecs == 0:
        logger.error("Loaded index is empty (0 vectors indexed).")
        return 1


    # 2. Load ID mapping
    id_mapping_path = cfg.index_dir / "id_mapping.parquet"
    if not id_mapping_path.is_file():
        logger.error("ID mapping table not found at %s.", id_mapping_path)
        return 1

    df_mapping = pd.read_parquet(id_mapping_path)
    mapping_lookup = df_mapping.set_index("faiss_id").to_dict(orient="index")

    # 3. Load landmark metadata lookup table
    logger.info("Loading landmark metadata lookup table...")
    lookup = LandmarkLookupTable()
    for split in ["train", "index"]:
        meta_p = cfg.processed_dir / f"metadata_{split}.parquet"
        if meta_p.is_file():
            lookup.load_from_parquet(meta_p)

    # 4. Initialize CLIP encoder
    logger.info("Initializing CLIP model (%s on %s)...", cfg.clip_model_name, args.device or cfg.device)
    encoder = CLIPEncoder(
        model_name=cfg.clip_model_name,
        pretrained=args.pretrained or cfg.clip_pretrained,
        device=args.device or cfg.device,
    )

    # 5. Generate query embedding
    query_start = time.time()
    if args.query_image:
        query_path = Path(args.query_image)
        if not query_path.is_file():
            logger.error("Query image not found: %s", query_path)
            return 1
        logger.info("Encoding query image: %s...", query_path.name)
        with Image.open(query_path) as img:
            rgb_img = img.convert("RGB")
            query_emb = encoder.encode_pil_images([rgb_img])[0]
        query_desc = f"Image: {query_path.name}"
    else:
        logger.info("Encoding query text: '%s'...", args.query_text)
        query_emb = encoder.encode_text(args.query_text)
        query_desc = f"Text prompt: '{args.query_text}'"

    emb_time = time.time() - query_start

    # 6. Search FAISS index
    search_k = args.rerank_candidates if args.rerank else (args.top_k + 5)
    search_start = time.time()
    fusion_meta = {}

    if args.index_type == "ensemble":
        logger.info("Querying Ensemble (IVF-PQ + HNSW-PQ) with Reciprocal Rank Fusion...")
        sc_ivf, ids_ivf = index_ivf.search(query_emb.reshape(1, -1), top_k=search_k)
        sc_hnsw, ids_hnsw = index_hnsw.search(query_emb.reshape(1, -1), top_k=search_k)
        from src.index.fusion import reciprocal_rank_fusion
        scores, faiss_ids, fusion_meta = reciprocal_rank_fusion(
            [("ivfpq", sc_ivf[0], ids_ivf[0]), ("hnswpq", sc_hnsw[0], ids_hnsw[0])],
            k=60,
            top_k=search_k,
        )
    elif args.index_type == "hnswpq":
        logger.info("Querying FAISS HNSW-PQ index (top_k=%d, efSearch=%d)...", search_k, args.ef_search)
        scores, faiss_ids = index.search(query_emb.reshape(1, -1), top_k=search_k)
        scores = scores[0]
        faiss_ids = faiss_ids[0]
    else:
        logger.info("Querying FAISS IVF-PQ index (top_k=%d, nprobe=%d)...", search_k, args.nprobe)
        scores, faiss_ids = index.search(query_emb.reshape(1, -1), top_k=search_k)
        scores = scores[0]
        faiss_ids = faiss_ids[0]

    search_time = time.time() - search_start

    # Optional Alpha-Query Expansion (α-QE)
    qe_time = 0.0
    if args.alpha_qe:
        try:
            from src.reranking.alpha_qe import AlphaQueryExpander
            t_qe_start = time.time()
            expander = AlphaQueryExpander(alpha=3.0, num_candidates=5)
            expanded_emb = expander.expand_query(
                query_emb=query_emb,
                candidate_ids=list(faiss_ids),
                candidate_scores=list(scores),
                index=index,
            )
            if args.index_type == "ensemble":
                sc_ivf_qe, ids_ivf_qe = index_ivf.search(expanded_emb.reshape(1, -1), top_k=search_k)
                sc_hnsw_qe, ids_hnsw_qe = index_hnsw.search(expanded_emb.reshape(1, -1), top_k=search_k)
                scores, faiss_ids, fusion_meta = reciprocal_rank_fusion(
                    [("ivfpq", sc_ivf_qe[0], ids_ivf_qe[0]), ("hnswpq", sc_hnsw_qe[0], ids_hnsw_qe[0])],
                    k=60,
                    top_k=search_k,
                )
            else:
                scores_qe, ids_qe = index.search(expanded_emb.reshape(1, -1), top_k=search_k)
                scores = scores_qe[0]
                faiss_ids = ids_qe[0]
            qe_time = time.time() - t_qe_start
            logger.info("Alpha-Query Expansion completed in %.2f ms", qe_time * 1000.0)
        except Exception as e:
            logger.warning("Alpha-Query Expansion failed: %s. Using coarse retrieval.", e)


    # 7. Optional two-stage rerank
    if args.rerank:
        reranker = TwoStageReranker(cfg.embedding_dir, df_mapping)
        valid_candidates = [fid for fid in faiss_ids if fid >= 0]
        rerank_results = reranker.rerank_candidates(query_emb, valid_candidates, top_k=args.top_k + 5)
        # Re-pack scores and IDs
        faiss_ids = np.array([r["faiss_id"] for r in rerank_results], dtype=np.int64)
        scores = np.array([r["exact_score"] for r in rerank_results], dtype=np.float32)

    # 8. Assemble retrieved matches and resolve landmark names
    retrieved_matches: List[Dict[str, Any]] = []
    rank = 1

    for fid, score in zip(faiss_ids, scores):
        if fid < 0:
            continue
        meta_row = mapping_lookup.get(fid, {})
        img_id = str(meta_row.get("image_id", ""))
        lid = str(meta_row.get("landmark_id", ""))
        split = str(meta_row.get("split", "train"))

        # Exclude query itself if requested
        if args.exclude_image_id and img_id == args.exclude_image_id:
            continue

        landmark_info = lookup.lookup_landmark(split, lid)
        image_info = lookup.lookup_image(img_id) or {}

        match_data = {
            "rank": rank,
            "faiss_id": int(fid),
            "score": round(float(score), 4),
            "image_id": img_id,
            "landmark_id": lid,
            "split": split,
            "landmark_name": landmark_info.landmark_name,
            "is_resolved": landmark_info.is_resolved,
            "category_url": landmark_info.category_url,
            "hierarchical_label": landmark_info.hierarchical_label,
            "image_path": image_info.get("full_path")
            or (
                str(cfg.image_dir / split / img_id[0] / img_id[1] / img_id[2] / f"{img_id}.jpg")
                if len(img_id) >= 3
                else ""
            ),
        }
        retrieved_matches.append(match_data)
        rank += 1
        if len(retrieved_matches) >= args.top_k:
            break

    # Optional Stage-2 Geometric Verification
    geom_time = 0.0
    if args.geometric_rerank:
        if not args.query_image:
            logger.warning("Geometric verification is only applicable to image queries (--query-image). Skipping.")
        else:
            try:
                from src.reranking.pipeline import GeometricReranker
                logger.info("Executing Stage-2 Geometric Verification reranker (matcher=%s)...", args.matcher)
                t_g_start = time.time()
                reranker = GeometricReranker(matcher_type=args.matcher)
                with Image.open(Path(args.query_image)) as q_img:
                    retrieved_matches = reranker.rerank(
                        query_image=q_img.convert("RGB"),
                        candidates=retrieved_matches,
                        top_k=args.top_k,
                    )
                geom_time = time.time() - t_g_start
                logger.info("Geometric verification (%s) finished in %.2f ms", args.matcher, geom_time * 1000.0)
            except ImportError:
                logger.warning("src.reranking module not found. Skipping geometric verification.")
            except Exception as e:
                logger.warning("Geometric verification encountered an error: %s. Using coarse ranks.", e)


    # 9. Perform similarity-weighted voting
    voting_result = compute_similarity_weighted_votes(retrieved_matches)

    # 10. Display Results
    print("\n" + "=" * 90)
    print(f" VISUAL LANDMARK SEARCH RESULTS ({query_desc})")
    timing_str = f"Latency: {search_time * 1000:.2f} ms search ({emb_time * 1000:.2f} ms embedding"
    if geom_time > 0:
        timing_str += f", {geom_time * 1000:.2f} ms geometric verification)"
    else:
        timing_str += ")"
    param_info = f"HNSW-PQ efSearch: {args.ef_search}" if args.index_type == "hnswpq" else f"IVF-PQ nprobe: {args.nprobe}"
    print(f" {param_info} | {timing_str}")
    print("=" * 90)

    print(f"\n{'Rank':<5} {'Score':<8} {'Landmark Name':<32} {'Landmark ID':<14} {'Image ID':<14} {'Split'}")
    print("-" * 90)
    for m in retrieved_matches:
        resolved_marker = "" if m["is_resolved"] else " [unresolved]"
        name_str = f"{m['landmark_name']}{resolved_marker}"[:31]
        score_val = m.get("final_score", m["score"])
        verified_tag = " [✓geom]" if m.get("geometric_verified") else ""
        print(f"{m['rank']:<5} {score_val:<8.4f} {name_str:<32} {m['landmark_id']:<14} {m['image_id']:<14} {m['split']}{verified_tag}")

    # Display Voting Summary
    print("\n" + "-" * 90)
    print(" SIMILARITY-WEIGHTED VOTING IDENTIFICATION")
    print("-" * 90)
    lead = voting_result.leading_candidate
    if lead:
        print(f"  Leading Candidate   : {lead.landmark_name} (Split: {lead.split}, ID: {lead.landmark_id})")
        print(f"  Total Vote Score    : {lead.total_vote_score:.4f} (from {lead.match_count} supporting image matches)")
        print(f"  Confidence Margin   : {voting_result.margin:.4f} over runner-up")
        if lead.category_url:
            print(f"  Category Source     : {lead.category_url}")
        if lead.hierarchical_label:
            print(f"  Hierarchy           : {lead.hierarchical_label}")

        supporting_ids = [s["image_id"] for s in lead.supporting_images]
        print(f"  Supporting Images   : {', '.join(supporting_ids[:5])}{' ...' if len(supporting_ids) > 5 else ''}")
    else:
        print("  No candidate landmarks retrieved.")

    print(f"\n  NOTE: {voting_result.note}")
    print("=" * 90 + "\n")

    # Export outputs if requested
    if args.output_csv:
        df_out = pd.DataFrame(retrieved_matches)
        df_out.to_csv(args.output_csv, index=False)
        logger.info("Saved search results to CSV: %s", args.output_csv)

    if args.output_json:
        export_payload = {
            "query": query_desc,
            "top_k": args.top_k,
            "nprobe": args.nprobe,
            "search_latency_ms": round(search_time * 1000, 2),
            "retrieved_matches": retrieved_matches,
            "voting_summary": {
                "leading_landmark": lead.landmark_name if lead else None,
                "leading_id": lead.landmark_id if lead else None,
                "total_vote_score": lead.total_vote_score if lead else 0.0,
                "match_count": lead.match_count if lead else 0,
                "margin": voting_result.margin,
            },
            "note": voting_result.note,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(export_payload, f, indent=2)
        logger.info("Saved search results to JSON: %s", args.output_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
