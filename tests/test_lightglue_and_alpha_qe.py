"""Unit Tests for Alpha-Query Expansion (α-QE) and LightGlue Neural Matcher."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.reranking.alpha_qe import AlphaQueryExpander
from src.reranking.lightglue_verifier import LightGlueVerifier
from src.reranking.pipeline import GeometricReranker


class MockIndex:
    """Mock vector index that supports reconstruct."""

    def __init__(self, vectors: dict):
        self.vectors = vectors

    def reconstruct(self, key: int) -> np.ndarray:
        if key in self.vectors:
            return self.vectors[key]
        raise KeyError(f"Vector {key} not found")


def create_synthetic_pattern_image(size=(256, 256)) -> Image.Image:
    """Generate high-contrast geometric pattern with rich local features."""
    img = Image.new("RGB", size, color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([30, 30, 110, 110], outline=(0, 0, 0), width=4, fill=(100, 150, 200))
    draw.ellipse([120, 40, 190, 110], outline=(200, 0, 0), width=3, fill=(50, 200, 50))
    draw.polygon([(40, 200), (100, 130), (160, 200)], outline=(0, 0, 150), fill=(255, 200, 50))
    for i in range(8):
        draw.line([(20 + i * 20, 210), (30 + i * 20, 240)], fill=(0, 0, 0), width=2)
    return img


def test_alpha_qe_expansion():
    """Verify that α-QE shifts the query toward high-scoring candidates."""
    # Target direction: [1, 0, 0]
    q = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    q = q / np.linalg.norm(q)

    # Candidate 1 is strongly aligned [1, 0, 0], candidate 2 is noise [0, 1, 0]
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    mock_idx = MockIndex({1: v1, 2: v2})

    expander = AlphaQueryExpander(alpha=3.0, num_candidates=2)
    expanded = expander.expand_query(
        query_emb=q,
        candidate_ids=[1, 2],
        candidate_scores=[0.9, 0.2],
        index=mock_idx,
    )

    # Expanded vector should be L2-normalized
    assert np.isclose(np.linalg.norm(expanded), 1.0, atol=1e-5)
    # Alignment with [1, 0, 0] should increase because candidate 1 had high score (0.9^3 = 0.729 vs 0.2^3 = 0.008)
    assert expanded[0] > q[0]


def test_lightglue_feature_extraction():
    """Verify that SuperPoint extracts coordinates and descriptors."""
    verifier = LightGlueVerifier(max_num_keypoints=256)
    img = create_synthetic_pattern_image()

    kpts, feats, shape = verifier.extract_features(img)
    assert len(kpts) >= 10
    assert "keypoints" in feats
    assert "descriptors" in feats
    assert shape == (256, 256)


def test_lightglue_matching_transformed_image():
    """Verify that rotated image produces high inliers with LightGlue."""
    verifier = LightGlueVerifier(max_num_keypoints=256, min_inliers=8)
    base_img = create_synthetic_pattern_image()
    transformed_img = base_img.rotate(10)

    res = verifier.verify_images(base_img, transformed_img)
    assert res.num_matches > 5
    assert res.num_inliers >= 8
    assert res.is_verified is True
    assert res.homography is not None


def test_lightglue_unrelated_images():
    """Verify that noise image results in 0 or near-0 matches with LightGlue."""
    verifier = LightGlueVerifier(max_num_keypoints=256, min_inliers=10)
    base_img = create_synthetic_pattern_image()
    noise_arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    noise_img = Image.fromarray(noise_arr)

    res = verifier.verify_images(base_img, noise_img)
    assert res.is_verified is False
    assert res.num_inliers < 10


def test_geometric_reranker_with_lightglue(tmp_path):
    """Test full GeometricReranker using matcher_type='lightglue'."""
    img1 = create_synthetic_pattern_image()
    img2 = img1.rotate(5)
    unrelated = Image.new("RGB", (256, 256), color=(20, 20, 20))

    p1 = tmp_path / "match.jpg"
    p2 = tmp_path / "unrelated.jpg"
    img2.save(p1)
    unrelated.save(p2)

    candidates = [
        {"faiss_id": 1, "image_id": "unrelated", "image_path": str(p2), "score": 0.85},
        {"faiss_id": 2, "image_id": "match", "image_path": str(p1), "score": 0.80},
    ]

    reranker = GeometricReranker(matcher_type="lightglue", verification_boost=1.5)
    reranked = reranker.rerank(query_image=img1, candidates=candidates)

    assert len(reranked) == 2
    assert reranked[0]["image_id"] == "match"
    assert reranked[0]["geometric_verified"] is True
    assert reranked[0]["rank"] == 1
