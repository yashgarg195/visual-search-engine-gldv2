"""Unit Tests for FAISS IVF-PQ Index, Validations, and Weighted Voting."""

import numpy as np
import pytest

from src.index.ivfpq import IVFPQIndex, validate_ivfpq_parameters
from src.index.voting import compute_similarity_weighted_votes


def test_ivfpq_parameter_validations():
    # 1. Dimension not divisible by m (512 % 7 != 0)
    with pytest.raises(ValueError, match="divisible by m"):
        validate_ivfpq_parameters(d=512, nlist=16, m=7, nbits=8, num_training_vectors=500)

    # 2. Training vectors smaller than nlist
    with pytest.raises(ValueError, match="smaller than nlist"):
        validate_ivfpq_parameters(d=512, nlist=1024, m=64, nbits=8, num_training_vectors=100)

    # 3. Training vectors smaller than 2^nbits (2^8 = 256)
    with pytest.raises(ValueError, match="smaller than 2\\^nbits"):
        validate_ivfpq_parameters(d=512, nlist=16, m=64, nbits=8, num_training_vectors=150)


def test_ivfpq_train_add_search(tmp_path):
    d = 64
    nlist = 8
    m = 8
    nbits = 8
    num_train = 300

    # Generate synthetic normalized vectors
    rng = np.random.RandomState(42)
    raw = rng.randn(num_train, d).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    vectors = raw / norms

    index = IVFPQIndex(d=d, nlist=nlist, m=m, nbits=nbits, nprobe=4)
    assert not index.is_trained

    # Train
    index.train(vectors)
    assert index.is_trained
    assert index.total_vectors == 0

    # Add with explicit int64 IDs
    ids = np.arange(1000, 1000 + num_train, dtype=np.int64)
    index.add(vectors, ids)
    assert index.total_vectors == num_train

    # Search with first vector
    scores, retrieved_ids = index.search(vectors[0:1], top_k=5)
    assert scores.shape == (1, 5)
    assert retrieved_ids.shape == (1, 5)
    # The first vector should be retrieved among top results
    assert 1000 in retrieved_ids[0]

    # Save and reload test
    index_file = tmp_path / "test_ivfpq.index"
    index.save(index_file)
    assert index_file.is_file()

    reloaded = IVFPQIndex(d=d)
    reloaded.load(index_file)
    assert reloaded.is_trained
    assert reloaded.total_vectors == num_train

    r_scores, r_ids = reloaded.search(vectors[0:1], top_k=5)
    assert np.array_equal(retrieved_ids, r_ids)


def test_similarity_weighted_voting():
    matches = [
        {
            "rank": 1,
            "score": 0.85,
            "split": "train",
            "landmark_id": "L1",
            "landmark_name": "Eiffel Tower",
            "image_id": "img1",
            "is_resolved": True,
        },
        {
            "rank": 2,
            "score": 0.82,
            "split": "train",
            "landmark_id": "L1",
            "landmark_name": "Eiffel Tower",
            "image_id": "img2",
            "is_resolved": True,
        },
        {
            "rank": 3,
            "score": 0.70,
            "split": "train",
            "landmark_id": "L2",
            "landmark_name": "Taj Mahal",
            "image_id": "img3",
            "is_resolved": True,
        },
    ]

    res = compute_similarity_weighted_votes(matches)
    assert res.leading_candidate is not None
    assert res.leading_candidate.landmark_name == "Eiffel Tower"
    # L1 vote = 0.85 + 0.82 = 1.67
    assert res.leading_candidate.total_vote_score == pytest.approx(1.67, abs=1e-3)
    assert res.leading_candidate.match_count == 2

    # L2 vote = 0.70
    assert res.second_candidate is not None
    assert res.second_candidate.landmark_name == "Taj Mahal"
    assert res.margin == pytest.approx(1.67 - 0.70, abs=1e-3)
