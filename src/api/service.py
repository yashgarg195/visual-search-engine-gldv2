"""Backend Search Engine Service for FastAPI."""

import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from PIL import Image

from src.config import get_config
from src.index.ivfpq import IVFPQIndex
from src.index.rerank import TwoStageReranker
from src.index.voting import compute_similarity_weighted_votes
from src.metadata.lookup import LandmarkLookupTable
from src.models.clip_encoder import CLIPEncoder
from src.utils.logging import get_logger

try:
    from src.reranking.pipeline import GeometricReranker
    HAS_GEOMETRIC_RERANKER = True
except ImportError:
    HAS_GEOMETRIC_RERANKER = False

logger = get_logger("search_service")


class SearchEngineService:
    """Manages model, index, and metadata for fast concurrent visual landmark retrieval."""

    def __init__(self, env_file: Optional[str] = None):
        self.cfg = get_config(env_file)
        self.encoder: Optional[CLIPEncoder] = None
        self.index: Optional[IVFPQIndex] = None
        self.mapping_lookup: Dict[int, Dict[str, Any]] = {}
        self.df_mapping: Optional[pd.DataFrame] = None
        self.lookup: Optional[LandmarkLookupTable] = None
        self.is_ready = False

    def initialize(self):
        """Load OpenCLIP, FAISS index, and lookup tables into memory once."""
        if self.is_ready:
            return

        logger.info("Initializing Search Engine Service...")

        # 1. Initialize OpenCLIP
        logger.info("Loading OpenCLIP model (%s)...", self.cfg.clip_model_name)
        self.encoder = CLIPEncoder(
            model_name=self.cfg.clip_model_name,
            pretrained=self.cfg.clip_pretrained,
            device=self.cfg.device,
        )

        # 2. Load FAISS IVF-PQ Index
        index_path = self.cfg.index_dir / "ivfpq_populated.index"
        if not index_path.is_file():
            index_path = self.cfg.index_dir / "ivfpq_trained.index"

        if not index_path.is_file():
            raise FileNotFoundError(f"FAISS index file not found in {self.cfg.index_dir}")

        logger.info("Loading FAISS index from %s...", index_path)
        self.index = IVFPQIndex()
        self.index.load(index_path)
        self.index.set_nprobe(self.cfg.faiss_nprobe)

        # 3. Load ID Mapping Table
        mapping_path = self.cfg.index_dir / "id_mapping.parquet"
        if mapping_path.is_file():
            logger.info("Loading ID mapping table from %s...", mapping_path)
            self.df_mapping = pd.read_parquet(mapping_path)
            self.mapping_lookup = self.df_mapping.set_index("faiss_id").to_dict(orient="index")
        else:
            logger.warning("ID mapping table not found at %s. Results will lack image ID resolution.", mapping_path)

        # 4. Load Landmark Metadata
        logger.info("Loading Landmark Lookup Table...")
        self.lookup = LandmarkLookupTable()
        for split in ["train", "index"]:
            meta_p = self.cfg.processed_dir / f"metadata_{split}.parquet"
            if meta_p.is_file():
                self.lookup.load_from_parquet(meta_p)

        self.is_ready = True
        logger.info(
            "Search Engine Service successfully initialized! Total indexed vectors: %d",
            self.index.total_vectors if self.index else 0,
        )

    def get_info(self) -> Dict[str, Any]:
        """Return system stats and index info."""
        return {
            "model_name": self.cfg.clip_model_name,
            "pretrained": self.cfg.clip_pretrained,
            "embedding_dim": self.encoder.embedding_dim if self.encoder else self.cfg.embedding_dim,
            "device": str(self.encoder.device) if self.encoder else "unknown",
            "total_vectors": self.index.total_vectors if self.index else 0,
            "default_nprobe": self.cfg.faiss_nprobe,
            "default_top_k": 10,
            "is_ready": self.is_ready,
        }

    def _execute_search(
        self,
        query_emb: np.ndarray,
        query_desc: str,
        top_k: int = 10,
        nprobe: int = 32,
        rerank: bool = False,
    ) -> Dict[str, Any]:
        """Core search and aggregation routine given a 512-dim normalized embedding."""
        if not self.is_ready or self.index is None:
            raise RuntimeError("Search service is not initialized.")

        self.index.set_nprobe(nprobe)
        search_k = max(top_k + 5, 20)

        t_search_start = time.perf_counter()
        scores, faiss_ids = self.index.search(query_emb.reshape(1, -1), top_k=search_k)
        t_search_ms = (time.perf_counter() - t_search_start) * 1000.0

        scores = scores[0]
        faiss_ids = faiss_ids[0]

        # Optional Two-stage exact rerank
        if rerank and self.df_mapping is not None:
            reranker = TwoStageReranker(self.cfg.embedding_dir, self.df_mapping)
            valid_ids = [fid for fid in faiss_ids if fid >= 0]
            reranked = reranker.rerank_candidates(query_emb, valid_ids, top_k=top_k + 5)
            faiss_ids = np.array([r["faiss_id"] for r in reranked], dtype=np.int64)
            scores = np.array([r["exact_score"] for r in reranked], dtype=np.float32)

        # Assemble retrieved records
        retrieved_matches: List[Dict[str, Any]] = []
        rank = 1

        for fid, score in zip(faiss_ids, scores):
            if fid < 0:
                continue

            meta_row = self.mapping_lookup.get(int(fid), {})
            img_id = str(meta_row.get("image_id", ""))
            lid = str(meta_row.get("landmark_id", ""))
            split = str(meta_row.get("split", "train"))

            landmark_info = self.lookup.lookup_landmark(split, lid) if self.lookup else None
            image_info = (self.lookup.lookup_image(img_id) or {}) if self.lookup else {}

            category_url = landmark_info.category_url if landmark_info else ""
            hierarchical_label = landmark_info.hierarchical_label if landmark_info else ""
            landmark_name = (
                landmark_info.landmark_name
                if (landmark_info and landmark_info.landmark_name)
                else f"Landmark #{lid}"
            )

            img_rel_path = f"{split}/{img_id[0]}/{img_id[1]}/{img_id[2]}/{img_id}.jpg" if len(img_id) >= 3 else ""
            img_full_path = str(self.cfg.image_dir / img_rel_path) if img_rel_path else ""

            match_data = {
                "rank": rank,
                "faiss_id": int(fid),
                "score": round(float(score), 4),
                "image_id": img_id,
                "landmark_id": lid,
                "split": split,
                "landmark_name": landmark_name,
                "is_resolved": bool(landmark_info.is_resolved) if landmark_info else False,
                "category_url": category_url,
                "hierarchical_label": hierarchical_label,
                "image_path": img_full_path,
            }
            retrieved_matches.append(match_data)
            rank += 1
            if len(retrieved_matches) >= top_k:
                break

        # Voting aggregation
        voting_result = compute_similarity_weighted_votes(retrieved_matches)
        voting_summary: Optional[Dict[str, Any]] = None

        if voting_result.leading_candidate:
            lead = voting_result.leading_candidate
            voting_summary = {
                "landmark_name": lead.landmark_name,
                "landmark_id": lead.landmark_id,
                "split": lead.split,
                "total_vote_score": round(lead.total_vote_score, 4),
                "confidence_margin": round(voting_result.margin, 4),
                "match_count": lead.match_count,
                "category_url": lead.category_url or "",
                "hierarchical_label": lead.hierarchical_label or "",
            }

        return {
            "query_desc": query_desc,
            "search_latency_ms": round(t_search_ms, 2),
            "top_k": top_k,
            "nprobe": nprobe,
            "voting_summary": voting_summary,
            "matches": retrieved_matches,
        }

    def search_text(
        self,
        query_text: str,
        top_k: int = 10,
        nprobe: int = 32,
        rerank: bool = False,
    ) -> Dict[str, Any]:
        """Encode natural language text query and retrieve matching landmarks."""
        if not self.encoder:
            raise RuntimeError("Encoder not initialized.")

        t_emb_start = time.perf_counter()
        query_emb = self.encoder.encode_text(query_text)
        t_emb_ms = (time.perf_counter() - t_emb_start) * 1000.0

        res = self._execute_search(
            query_emb=query_emb,
            query_desc=f"Text: '{query_text}'",
            top_k=top_k,
            nprobe=nprobe,
            rerank=rerank,
        )
        res["query_type"] = "text"
        res["query_input"] = query_text
        res["embedding_latency_ms"] = round(t_emb_ms, 2)
        res["total_latency_ms"] = round(t_emb_ms + res["search_latency_ms"], 2)
        return res

    def search_image(
        self,
        image_bytes: bytes,
        filename: str = "uploaded_image.jpg",
        top_k: int = 10,
        nprobe: int = 32,
        rerank: bool = False,
        rerank_geometric: bool = False,
    ) -> Dict[str, Any]:
        """Encode raw image bytes into OpenCLIP representation and retrieve matching landmarks."""
        if not self.encoder:
            raise RuntimeError("Encoder not initialized.")

        t_emb_start = time.perf_counter()
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            rgb_img = pil_img.convert("RGB")
            query_emb = self.encoder.encode_pil_images([rgb_img])[0]
        t_emb_ms = (time.perf_counter() - t_emb_start) * 1000.0

        res = self._execute_search(
            query_emb=query_emb,
            query_desc=f"Image: {filename}",
            top_k=top_k,
            nprobe=nprobe,
            rerank=rerank,
        )

        # Optional Stage-2 Geometric Verification
        t_geom_ms = 0.0
        if rerank_geometric and HAS_GEOMETRIC_RERANKER:
            t_geom_start = time.perf_counter()
            reranker = GeometricReranker()
            res["matches"] = reranker.rerank(
                query_image=rgb_img,
                candidates=res["matches"],
                top_k=top_k,
            )
            # Recompute voting summary with reranked matches
            voting_res = compute_similarity_weighted_votes(res["matches"])
            if voting_res.leading_candidate:
                lead = voting_res.leading_candidate
                res["voting_summary"] = {
                    "landmark_name": lead.landmark_name,
                    "landmark_id": lead.landmark_id,
                    "split": lead.split,
                    "total_vote_score": round(lead.total_vote_score, 4),
                    "confidence_margin": round(voting_res.margin, 4),
                    "match_count": lead.match_count,
                    "category_url": lead.category_url or "",
                    "hierarchical_label": lead.hierarchical_label or "",
                }
            t_geom_ms = (time.perf_counter() - t_geom_start) * 1000.0
            res["geometric_rerank_ms"] = round(t_geom_ms, 2)

        res["query_type"] = "image"
        res["query_input"] = filename
        res["embedding_latency_ms"] = round(t_emb_ms, 2)
        res["total_latency_ms"] = round(t_emb_ms + res["search_latency_ms"] + t_geom_ms, 2)
        return res
