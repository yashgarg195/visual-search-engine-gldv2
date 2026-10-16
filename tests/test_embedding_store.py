"""Unit Tests for EmbeddingStore and Resumption Logic."""

import numpy as np
import pandas as pd
import pytest

from src.storage.embedding_store import EmbeddingStore


def test_embedding_store_write_and_resume(tmp_path):
    store = EmbeddingStore(tmp_path)
    store.set_model_metadata("ViT-B-32", "openai", 512)

    assert store.get_next_shard_index() == 0
    assert len(store.get_completed_image_ids()) == 0

    # Write shard 0
    emb0 = np.random.randn(10, 512).astype(np.float32)
    meta0 = pd.DataFrame({
        "image_id": [f"img_{i:04d}" for i in range(10)],
        "landmark_id": ["L1"] * 10,
        "split": ["train"] * 10,
    })
    store.write_shard(0, emb0, meta0)

    completed = store.get_completed_image_ids()
    assert len(completed) == 10
    assert "img_0000" in completed
    assert "img_0009" in completed
    assert store.get_next_shard_index() == 1

    # Write shard 1
    emb1 = np.random.randn(5, 512).astype(np.float32)
    meta1 = pd.DataFrame({
        "image_id": [f"img_{i:04d}" for i in range(10, 15)],
        "landmark_id": ["L2"] * 5,
        "split": ["train"] * 5,
    })
    store.write_shard(1, emb1, meta1)

    completed_updated = store.get_completed_image_ids()
    assert len(completed_updated) == 15
    assert store._manifest["total_vectors"] == 15


def test_embedding_store_alignment_check(tmp_path):
    store = EmbeddingStore(tmp_path)
    store.set_model_metadata("ViT-B-32", "openai", 512)

    emb = np.random.randn(10, 512).astype(np.float32)
    # Mismatched rows count: 8 rows vs 10 vectors
    meta_mismatch = pd.DataFrame({
        "image_id": [f"img_{i}" for i in range(8)],
        "landmark_id": ["L1"] * 8,
    })

    with pytest.raises(ValueError, match="Length mismatch"):
        store.write_shard(0, emb, meta_mismatch)


def test_embedding_store_sampling(tmp_path):
    store = EmbeddingStore(tmp_path)
    store.set_model_metadata("ViT-B-32", "openai", 512)

    emb = np.random.randn(50, 512).astype(np.float32)
    meta = pd.DataFrame({
        "image_id": [f"id_{i}" for i in range(50)],
        "landmark_id": ["L1"] * 50,
    })
    store.write_shard(0, emb, meta)

    sample = store.sample_training_vectors(max_samples=20, seed=42)
    assert sample.shape == (20, 512)
    assert sample.dtype == np.float32
