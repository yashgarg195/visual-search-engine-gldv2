"""Unit Tests for Stage-2 Geometric Verification & Reranking Module."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.reranking.geometric_verifier import GeometricVerifier, GeometricVerificationResult
from src.reranking.pipeline import GeometricReranker


def create_synthetic_pattern_image(size=(300, 300)) -> Image.Image:
    """Generate a high-contrast geometric pattern with rich SIFT keypoints."""
    img = Image.new("RGB", size, color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    # Draw geometric shapes and text-like patterns
    draw.rectangle([30, 30, 120, 120], outline=(0, 0, 0), width=4, fill=(100, 150, 200))
    draw.ellipse([140, 40, 220, 120], outline=(200, 0, 0), width=3, fill=(50, 200, 50))
    draw.polygon([(50, 250), (120, 160), (190, 250)], outline=(0, 0, 150), fill=(255, 200, 50))
    for i in range(10):
        draw.line([(20 + i * 25, 260), (30 + i * 25, 290)], fill=(0, 0, 0), width=2)
        draw.rectangle([210, 140 + i * 15, 280, 150 + i * 15], fill=(30 * i, 20 * i, 10 * i))

    return img


def test_geometric_verifier_feature_extraction():
    """Verify that SIFT extracts coordinates and descriptors properly."""
    verifier = GeometricVerifier(detector_type="sift", max_features=500)
    img = create_synthetic_pattern_image()

    kpts, descs, shape = verifier.extract_features(img)
    assert len(kpts) > 20, f"Expected >20 keypoints, found {len(kpts)}"
    assert descs.shape[0] == len(kpts)
    assert descs.shape[1] == 128
    assert shape == (300, 300)


def test_geometric_verifier_matching_and_transformation():
    """Verify that an image and its cropped/rotated copy produce high inliers."""
    verifier = GeometricVerifier(detector_type="sift", min_inliers=8)
    base_img = create_synthetic_pattern_image()

    # Create slightly translated and rotated version
    transformed_img = base_img.rotate(10, expand=False).crop((10, 10, 290, 290))

    result: GeometricVerificationResult = verifier.verify_images(base_img, transformed_img)
    assert result.num_matches > 10
    assert result.num_inliers >= 8
    assert result.inlier_ratio > 0.3
    assert result.is_verified is True
    assert result.homography is not None


def test_geometric_verifier_unrelated_images():
    """Verify that pure noise or totally unrelated images fail verification."""
    verifier = GeometricVerifier(detector_type="sift", min_inliers=12)
    img1 = create_synthetic_pattern_image()
    # Random noise image
    noise_arr = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    img2 = Image.fromarray(noise_arr)

    result = verifier.verify_images(img1, img2)
    assert result.is_verified is False
    assert result.num_inliers < 12


def test_geometric_reranker_pipeline(tmp_path):
    """Test full Stage-2 reranking pipeline re-orders candidate matches."""
    img1 = create_synthetic_pattern_image()
    img2 = img1.rotate(5, expand=False)
    unrelated = Image.new("RGB", (300, 300), color=(10, 10, 10))

    # Save images to temp paths
    p1 = tmp_path / "match.jpg"
    p2 = tmp_path / "unrelated.jpg"
    img2.save(p1)
    unrelated.save(p2)

    # Simulated coarse retrieval results where unrelated had slightly higher coarse score
    candidates = [
        {"faiss_id": 1, "image_id": "unrelated", "image_path": str(p2), "score": 0.85},
        {"faiss_id": 2, "image_id": "match", "image_path": str(p1), "score": 0.80},
    ]

    reranker = GeometricReranker(alpha=0.4)
    reranked = reranker.rerank(query_image=img1, candidates=candidates)

    # The geometrically verified image should be boosted to rank 1
    assert len(reranked) == 2
    assert reranked[0]["image_id"] == "match"
    assert reranked[0]["geometric_verified"] is True
    assert reranked[0]["rank"] == 1


def test_geometric_reranker_handles_none_query():
    """Verify that a None query image leaves candidates untouched."""
    candidates = [
        {"faiss_id": 1, "score": 0.9},
        {"faiss_id": 2, "score": 0.8},
    ]
    reranker = GeometricReranker()
    res = reranker.rerank(query_image=None, candidates=candidates)
    assert res == candidates
