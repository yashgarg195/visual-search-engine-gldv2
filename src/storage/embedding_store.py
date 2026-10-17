"""Persistent Sharded Embedding Store with Atomic Manifest and Resumption Support."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.utils.io import atomic_save_json, atomic_save_numpy, atomic_save_parquet, load_json
from src.utils.logging import get_logger

logger = get_logger("embedding_store")


class EmbeddingStore:
    """Manages sharded embedding files, companion metadata, and the global manifest."""

    MANIFEST_FILENAME = "embedding_manifest.json"

    def __init__(self, store_dir: Path):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.store_dir / self.MANIFEST_FILENAME
        self._manifest: Dict[str, Any] = self._load_or_init_manifest()

    def _load_or_init_manifest(self) -> Dict[str, Any]:
        """Load existing manifest or initialize a new schema."""
        if self.manifest_path.is_file():
            try:
                manifest = load_json(self.manifest_path)
                logger.info("Loaded existing embedding manifest (%d shards)", len(manifest.get("shards", [])))
                return manifest
            except Exception as e:
                logger.warning("Failed to parse existing manifest at %s: %s. Creating new one.", self.manifest_path, e)

        return {
            "model_name": None,
            "pretrained": None,
            "embedding_dim": None,
            "normalized": True,
            "shards": [],
            "total_vectors": 0,
            "total_failed": 0,
            "checkpoint_status": "in_progress",
        }

    @property
    def total_vectors(self) -> int:
        """Total number of vectors indexed across all shards."""
        return self._manifest.get("total_vectors", 0)

    @property
    def total_shards(self) -> int:
        """Total number of persistent shards."""
        return len(self._manifest.get("shards", []))

    @property
    def manifest(self) -> Dict[str, Any]:
        """Underlying manifest dictionary."""
        return self._manifest

    def save_manifest(self) -> None:
        """Persist manifest atomically."""
        atomic_save_json(self._manifest, self.manifest_path)

    def set_model_metadata(self, model_name: str, pretrained: str, embedding_dim: int) -> None:
        """Record model architecture details in manifest, verifying compatibility if exists."""
        if self._manifest["model_name"] and self._manifest["model_name"] != model_name:
            raise ValueError(
                f"Store model mismatch: existing={self._manifest['model_name']}, new={model_name}"
            )
        if self._manifest["embedding_dim"] and self._manifest["embedding_dim"] != embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: existing={self._manifest['embedding_dim']}, new={embedding_dim}"
            )

        self._manifest["model_name"] = model_name
        self._manifest["pretrained"] = pretrained
        self._manifest["embedding_dim"] = embedding_dim
        self.save_manifest()

    def get_completed_image_ids(self) -> Set[str]:
        """Scan companion shard metadata to return the exact set of already-indexed image IDs."""
        completed_ids: Set[str] = set()
        for shard_info in self._manifest.get("shards", []):
            meta_rel = shard_info.get("meta_file")
            if not meta_rel:
                continue
            meta_path = self.store_dir / meta_rel
            if meta_path.is_file():
                try:
                    df = pd.read_parquet(meta_path, columns=["image_id"])
                    completed_ids.update(df["image_id"].astype(str).tolist())
                except Exception as e:
                    logger.warning("Error reading shard metadata %s: %s", meta_path, e)
        return completed_ids

    def write_shard(
        self,
        shard_idx: int,
        embeddings: np.ndarray,
        metadata_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Atomically persist an embedding array and companion metadata parquet shard.

        Ensures row order is strictly synchronized: row i of metadata corresponds
        to embeddings[i].
        """
        if len(embeddings) != len(metadata_df):
            raise ValueError(
                f"Length mismatch: {len(embeddings)} embeddings vs {len(metadata_df)} metadata rows!"
            )

        shard_name = f"shard_{shard_idx:05d}"
        emb_file = f"{shard_name}_emb.npy"
        meta_file = f"{shard_name}_meta.parquet"

        emb_path = self.store_dir / emb_file
        meta_path = self.store_dir / meta_file

        # 1. Atomic write of numpy vectors
        atomic_save_numpy(embeddings, emb_path)

        # 2. Atomic write of companion metadata
        # Ensure row_idx is recorded explicitly
        meta_df = metadata_df.copy()
        meta_df["shard_name"] = shard_name
        meta_df["shard_row_idx"] = np.arange(len(meta_df), dtype=np.int64)
        atomic_save_parquet(meta_df, meta_path)

        # 3. Update manifest
        shard_info = {
            "shard_id": shard_idx,
            "shard_name": shard_name,
            "emb_file": emb_file,
            "meta_file": meta_file,
            "num_vectors": len(embeddings),
            "first_id": str(meta_df["image_id"].iloc[0]),
            "last_id": str(meta_df["image_id"].iloc[-1]),
        }

        # Replace or append shard entry
        shards = [s for s in self._manifest.get("shards", []) if s["shard_id"] != shard_idx]
        shards.append(shard_info)
        shards.sort(key=lambda s: s["shard_id"])
        self._manifest["shards"] = shards

        self._manifest["total_vectors"] = sum(s["num_vectors"] for s in shards)
        self.save_manifest()

        logger.info(
            "Wrote shard %s (%d vectors, shape %s) to %s",
            shard_name,
            len(embeddings),
            embeddings.shape,
            self.store_dir,
        )
        return shard_info

    def get_next_shard_index(self) -> int:
        """Return the next available sequential shard index."""
        shards = self._manifest.get("shards", [])
        if not shards:
            return 0
        return max(s["shard_id"] for s in shards) + 1

    def iter_shards(self, mmap: bool = True):
        """Generator yielding (shard_info, embeddings_array, metadata_df) one shard at a time."""
        for shard_info in self._manifest.get("shards", []):
            emb_path = self.store_dir / shard_info["emb_file"]
            meta_path = self.store_dir / shard_info["meta_file"]

            if not emb_path.is_file() or not meta_path.is_file():
                logger.error("Missing shard files for %s", shard_info["shard_name"])
                continue

            mmap_mode = "r" if mmap else None
            embeddings = np.load(emb_path, mmap_mode=mmap_mode)
            meta_df = pd.read_parquet(meta_path)

            yield shard_info, embeddings, meta_df

    def sample_training_vectors(self, max_samples: int, seed: int = 42) -> np.ndarray:
        """Sample a representative subset of embeddings across all shards for index training."""
        total_vectors = self._manifest.get("total_vectors", 0)
        if total_vectors == 0:
            raise ValueError("No embeddings found in store for training.")

        shards = self._manifest.get("shards", [])
        rng = np.random.RandomState(seed)

        # Allocate budget proportionally across shards
        samples_per_shard = []
        for s in shards:
            ratio = s["num_vectors"] / total_vectors
            take = max(1, int(round(ratio * max_samples)))
            samples_per_shard.append(min(take, s["num_vectors"]))

        gathered_arrays = []
        for s, n_take in zip(shards, samples_per_shard):
            emb_path = self.store_dir / s["emb_file"]
            arr = np.load(emb_path, mmap_mode="r")
            if n_take >= len(arr):
                gathered_arrays.append(np.array(arr, copy=True))
            else:
                idx = rng.choice(len(arr), size=n_take, replace=False)
                idx.sort()
                gathered_arrays.append(np.array(arr[idx], copy=True))

        combined = np.vstack(gathered_arrays)
        if len(combined) > max_samples:
            sub_idx = rng.choice(len(combined), size=max_samples, replace=False)
            combined = combined[sub_idx]

        logger.info(
            "Sampled %d vectors (dim=%d) across %d shards for training.",
            len(combined),
            combined.shape[1],
            len(shards),
        )
        return combined.astype(np.float32)
