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
from src.index.hnsw_pq import HNSWPQIndex
from src.index.rerank import TwoStageReranker
from src.index.voting import compute_similarity_weighted_votes
from src.metadata.lookup import LandmarkLookupTable
from src.models.clip_encoder import CLIPEncoder
from src.utils.logging import get_logger

from src.llm.location_resolver import LocationResolver
from src.reranking.reference_fetcher import ReferenceImageFetcher
from src.reranking.hybrid_fusion import HybridMultiSignalReranker


logger = get_logger("search_service")


class SearchEngineService:
    """Manages model, index, and metadata for fast concurrent visual landmark retrieval."""

    def __init__(self, env_file: Optional[str] = None):
        self.cfg = get_config(env_file)
        self.encoder: Optional[CLIPEncoder] = None
        self.index_ivf: Optional[IVFPQIndex] = None
        self.index_hnsw: Optional[HNSWPQIndex] = None
        self.index: Optional[IVFPQIndex] = None
        self.mapping_lookup: Dict[int, Dict[str, Any]] = {}
        self.df_mapping: Optional[pd.DataFrame] = None
        self.lookup: Optional[LandmarkLookupTable] = None
        self.is_ready = False

    def initialize(self):
        """Load OpenCLIP, FAISS indices, and lookup tables into memory once."""
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

        if index_path.is_file():
            logger.info("Loading FAISS IVF-PQ index from %s...", index_path)
            self.index_ivf = IVFPQIndex()
            self.index_ivf.load(index_path)
            self.index_ivf.set_nprobe(self.cfg.faiss_nprobe)
            self.index = self.index_ivf
        else:
            logger.warning("FAISS IVF-PQ index file not found in %s", self.cfg.index_dir)

        # 2b. Load FAISS HNSW-PQ Index if available
        self._ensure_hnsw_loaded()

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

        # 5. Initialize Location Resolver, Reference Fetcher, and Hybrid Fusion Reranker
        logger.info("Initializing Location Resolver, Reference Fetcher, and Hybrid Fusion Reranker...")
        self.location_resolver = LocationResolver()
        self.reference_fetcher = ReferenceImageFetcher()
        self.fusion_reranker = HybridMultiSignalReranker()

        self.is_ready = True
        logger.info(
            "Search Engine Service successfully initialized! Total IVF vectors: %d | Total HNSW vectors: %d",
            self.index_ivf.total_vectors if self.index_ivf else 0,
            self.index_hnsw.total_vectors if self.index_hnsw else 0,
        )

    def _ensure_hnsw_loaded(self) -> bool:
        """Dynamically load or reload HNSW-PQ index if a newer file exists."""
        hnsw_path = self.cfg.index_dir / "hnswpq_populated.index"
        if not hnsw_path.is_file():
            return False

        try:
            curr_mtime = hnsw_path.stat().st_mtime
            if self.index_hnsw is not None and getattr(self, "_hnsw_mtime", 0.0) == curr_mtime:
                return True

            logger.info("Loading / Reloading FAISS HNSW-PQ index from %s...", hnsw_path)
            new_hnsw = HNSWPQIndex()
            new_hnsw.load(hnsw_path)
            new_hnsw.set_ef_search(128)
            self.index_hnsw = new_hnsw
            self._hnsw_mtime = curr_mtime
            logger.info("Successfully loaded HNSW-PQ index with %d vectors (M=64, efSearch=128)", self.index_hnsw.total_vectors)
            return True
        except Exception as e:
            logger.warning("Failed to load HNSW-PQ index from %s: %s", hnsw_path, e)
            return self.index_hnsw is not None

    def get_info(self) -> Dict[str, Any]:
        """Return system stats and index info."""
        self._ensure_hnsw_loaded()
        available_indices = []
        if self.index_ivf is not None:
            available_indices.append("ivfpq")
        if self.index_hnsw is not None:
            available_indices.append("hnswpq")
        if self.index_ivf is not None and self.index_hnsw is not None:
            available_indices.append("ensemble")

        total_vecs = self.index_ivf.total_vectors if self.index_ivf else (self.index_hnsw.total_vectors if self.index_hnsw else 0)

        return {
            "model_name": self.cfg.clip_model_name,
            "pretrained": self.cfg.clip_pretrained,
            "embedding_dim": self.encoder.embedding_dim if self.encoder else self.cfg.embedding_dim,
            "device": str(self.encoder.device) if self.encoder else "unknown",
            "total_vectors": total_vecs,
            "ivf_vectors": self.index_ivf.total_vectors if self.index_ivf else 0,
            "hnsw_vectors": self.index_hnsw.total_vectors if self.index_hnsw else 0,
            "available_indices": available_indices,
            "default_index": "ivfpq",
            "default_nprobe": self.cfg.faiss_nprobe,
            "default_ef_search": 64,
            "default_top_k": 10,
            "is_ready": self.is_ready,
        }

    def _execute_search(
        self,
        query_emb: np.ndarray,
        query_desc: str,
        top_k: int = 10,
        nprobe: int = 64,
        index_type: str = "ivfpq",
        ef_search: int = 64,
        rerank: bool = False,
        alpha_qe: bool = False,
        query_image_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Core search and aggregation routine given a 512-dim normalized embedding."""
        if not self.is_ready:
            raise RuntimeError("Search service is not initialized.")

        is_ensemble = "ensemble" in index_type.lower() or "hybrid" in index_type.lower()
        is_hnsw = "hnsw" in index_type.lower() and not is_ensemble

        search_k = max(top_k + 5, 20)
        fusion_meta: Dict[int, Dict[str, Any]] = {}

        if is_ensemble:
            self._ensure_hnsw_loaded()
            if self.index_ivf is None or self.index_hnsw is None:
                raise RuntimeError("Ensemble search requires both IVF-PQ and HNSW-PQ indices to be available.")
            self.index_ivf.set_nprobe(nprobe or 64)
            self.index_hnsw.set_ef_search(ef_search or 64)

            t_search_start = time.perf_counter()
            sc_ivf, ids_ivf = self.index_ivf.search(query_emb.reshape(1, -1), top_k=search_k)
            sc_hnsw, ids_hnsw = self.index_hnsw.search(query_emb.reshape(1, -1), top_k=search_k)
            from src.index.fusion import reciprocal_rank_fusion
            scores, faiss_ids, fusion_meta = reciprocal_rank_fusion(
                [("ivfpq", sc_ivf[0], ids_ivf[0]), ("hnswpq", sc_hnsw[0], ids_hnsw[0])],
                k=60,
                top_k=search_k,
            )
            t_search_ms = (time.perf_counter() - t_search_start) * 1000.0
            active_index = self.index_hnsw
        elif is_hnsw:
            self._ensure_hnsw_loaded()
            if self.index_hnsw is None:
                raise RuntimeError("HNSW-PQ index is not available. Please train it first or select IVF-PQ.")
            active_index = self.index_hnsw
            active_index.set_ef_search(ef_search or 64)

            t_search_start = time.perf_counter()
            scores, faiss_ids = active_index.search(query_emb.reshape(1, -1), top_k=search_k)
            t_search_ms = (time.perf_counter() - t_search_start) * 1000.0
            scores = scores[0]
            faiss_ids = faiss_ids[0]
        else:
            if self.index_ivf is None:
                raise RuntimeError("IVF-PQ index is not initialized.")
            active_index = self.index_ivf
            active_index.set_nprobe(nprobe or 64)

            t_search_start = time.perf_counter()
            scores, faiss_ids = active_index.search(query_emb.reshape(1, -1), top_k=search_k)
            t_search_ms = (time.perf_counter() - t_search_start) * 1000.0
            scores = scores[0]
            faiss_ids = faiss_ids[0]

        # Optional Alpha-Query Expansion (α-QE)
        t_qe_ms = 0.0
        if alpha_qe and HAS_ALPHA_QE:
            t_qe_start = time.perf_counter()
            expander = AlphaQueryExpander(alpha=3.0, num_candidates=5)
            expanded_emb = expander.expand_query(
                query_emb=query_emb,
                candidate_ids=list(faiss_ids),
                candidate_scores=list(scores),
                index=active_index,
            )
            if is_ensemble:
                sc_ivf_qe, ids_ivf_qe = self.index_ivf.search(expanded_emb.reshape(1, -1), top_k=search_k)
                sc_hnsw_qe, ids_hnsw_qe = self.index_hnsw.search(expanded_emb.reshape(1, -1), top_k=search_k)
                from src.index.fusion import reciprocal_rank_fusion
                scores, faiss_ids, fusion_meta = reciprocal_rank_fusion(
                    [("ivfpq", sc_ivf_qe[0], ids_ivf_qe[0]), ("hnswpq", sc_hnsw_qe[0], ids_hnsw_qe[0])],
                    k=60,
                    top_k=search_k,
                )
            else:
                scores_qe, faiss_ids_qe = active_index.search(expanded_emb.reshape(1, -1), top_k=search_k)
                scores = scores_qe[0]
                faiss_ids = faiss_ids_qe[0]
            t_qe_ms = (time.perf_counter() - t_qe_start) * 1000.0

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
                "latitude": image_info.get("latitude"),
                "longitude": image_info.get("longitude"),
            }
            if int(fid) in fusion_meta:
                f_info = fusion_meta[int(fid)]
                match_data["consensus"] = f_info.get("consensus", False)
                match_data["sources"] = f_info.get("sources", [])
                match_data["rrf_score"] = f_info.get("rrf_score", 0.0)

            retrieved_matches.append(match_data)
            rank += 1
            if len(retrieved_matches) >= top_k:
                break


        # 6. Location Resolution & Canonical Reference Images
        location_info = None
        reference_images: List[Dict[str, Any]] = []
        if retrieved_matches:
            try:
                loc_res = self.location_resolver.resolve(query_desc, retrieved_matches)
                if loc_res:
                    location_info = loc_res.to_dict()
                    raw_refs = self.reference_fetcher.fetch_reference_images(loc_res.search_query, max_images=4)
                    reference_images = raw_refs
            except Exception as e:
                logger.warning("Location resolution / reference pipeline failed: %s", e)

        # 7. Hybrid Multi-Signal Fusion Reranking (50% Visual + 30% Vicinity Consensus + 20% Spatial Proximity)
        t_fusion_start = time.perf_counter()
        prominent_anchor = location_info.get("prominent_attraction") if location_info else None
        anchor_coords = None
        if location_info and location_info.get("latitude") and location_info.get("longitude"):
            try:
                anchor_coords = (float(location_info["latitude"]), float(location_info["longitude"]))
            except (ValueError, TypeError):
                anchor_coords = None

        final_matches = self.fusion_reranker.rerank(
            retrieved_matches=retrieved_matches,
            prominent_anchor=prominent_anchor,
            anchor_coords=anchor_coords,
        )
        t_fusion_ms = (time.perf_counter() - t_fusion_start) * 1000.0

        # Voting aggregation based on fused candidates
        voting_result = compute_similarity_weighted_votes(final_matches)
        voting_summary: Optional[Dict[str, Any]] = None

        if voting_result.leading_candidate:
            lead = voting_result.leading_candidate
            display_name = prominent_anchor if (prominent_anchor and location_info) else lead.landmark_name
            voting_summary = {
                "landmark_name": display_name,
                "specific_landmark": lead.landmark_name,
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
            "fusion_latency_ms": round(t_fusion_ms, 2),
            "top_k": top_k,
            "nprobe": nprobe,
            "ef_search": ef_search,
            "index_type": "ensemble" if is_ensemble else ("hnswpq" if is_hnsw else "ivfpq"),
            "voting_summary": voting_summary,
            "location_info": location_info,
            "reference_images": reference_images,
            "matches": final_matches[:top_k],
        }

    def search_text(
        self,
        query_text: str,
        top_k: int = 10,
        nprobe: int = 64,
        index_type: str = "ivfpq",
        ef_search: int = 64,
        rerank: bool = False,
        alpha_qe: bool = False,
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
            index_type=index_type,
            ef_search=ef_search,
            rerank=rerank,
            alpha_qe=alpha_qe,
        )
        res["query_type"] = "text"
        res["query_input"] = query_text
        res["embedding_latency_ms"] = round(t_emb_ms, 2)
        total_lat = t_emb_ms + res["search_latency_ms"] + res.get("alpha_qe_ms", 0.0)
        res["total_latency_ms"] = round(total_lat, 2)
        return res

    def search_image(
        self,
        image_bytes: bytes,
        filename: str = "uploaded_image.jpg",
        top_k: int = 10,
        nprobe: int = 64,
        index_type: str = "ivfpq",
        ef_search: int = 64,
        rerank: bool = False,
        alpha_qe: bool = False,
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
            index_type=index_type,
            ef_search=ef_search,
            rerank=rerank,
            alpha_qe=alpha_qe,
        )

        res["query_type"] = "image"
        res["query_input"] = filename
        res["embedding_latency_ms"] = round(t_emb_ms, 2)
        total_lat = t_emb_ms + res["search_latency_ms"] + res.get("fusion_latency_ms", 0.0)
        res["total_latency_ms"] = round(total_lat, 2)
        return res

