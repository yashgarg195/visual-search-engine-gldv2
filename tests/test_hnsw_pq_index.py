"""Unit Tests for FAISS HNSW-PQ Index Implementation."""

import numpy as np
import pytest
import faiss

from src.index.hnsw_pq import HNSWPQIndex, validate_hnsw_pq_parameters


def test_hnsw_pq_parameter_validation():
    """Verify mathematical and FAISS constraint validations."""
    # Invalid dimension divisibility (512 not divisible by 25)
    with pytest.raises(ValueError, match="divisible by m"):
        validate_hnsw_pq_parameters(d=512, m=25, M=32, num_training_vectors=1000)

    # Invalid M bounds (must be between 4 and 128)
    with pytest.raises(ValueError, match="between 4 and 128"):
        validate_hnsw_pq_parameters(d=512, m=64, M=2, num_training_vectors=1000)

    # Insufficient training points
    with pytest.raises(ValueError, match="smaller than 256"):
        validate_hnsw_pq_parameters(d=512, m=64, M=32, num_training_vectors=100)


def test_hnsw_pq_train_add_search(tmp_path):
    """Verify training, explicit ID mapping, persistence, and recall on vectors."""
    d = 64
    m = 8
    M = 16
    num_train = 10000
    num_add = 100

    np.random.seed(42)
    x_train = np.random.randn(num_train, d).astype(np.float32)
    faiss.normalize_L2(x_train)

    index = HNSWPQIndex(d=d, m=m, M=M, ef_construction=16, ef_search=16)
    assert not index.is_trained
    assert index.total_vectors == 0

    index.train(x_train)
    assert index.is_trained

    x_add = x_train[:num_add]
    ids = np.arange(1000, 1000 + num_add, dtype=np.int64)

    index.add(x_add, ids)
    assert index.total_vectors == num_add

    # Search: query vector should retrieve itself as top match
    scores, retrieved_ids = index.search(x_add[:5], top_k=3)
    assert scores.shape == (5, 3)
    assert retrieved_ids.shape == (5, 3)

    for i in range(5):
        assert ids[i] in retrieved_ids[i]

    # Test saving and loading
    save_path = tmp_path / "test_hnswpq.index"
    index.save(save_path)
    assert save_path.is_file()

    loaded = HNSWPQIndex()
    loaded.load(save_path)
    assert loaded.is_trained
    assert loaded.total_vectors == num_add

    l_scores, l_ids = loaded.search(x_add[:5], top_k=3)
    np.testing.assert_array_equal(retrieved_ids, l_ids)


def test_hnsw_pq_ef_search():
    """Verify runtime dynamic tuning of efSearch."""
    index = HNSWPQIndex(d=64, m=8, M=16)
    index.build_empty()
    assert index.ef_search == 64

    index.set_ef_search(128)
    assert index.ef_search == 128

    with pytest.raises(ValueError, match="ef_search must be >= 1"):
        index.set_ef_search(0)
