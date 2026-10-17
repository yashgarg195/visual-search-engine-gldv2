"""Modular Feature Matching and Geometric Verification for Stage-2 Reranking.

Implements SIFT/ORB keypoint extraction, Lowe's ratio test matching,
and robust geometric transformation estimation using USAC_MAGSAC.
Can be completely decoupled or removed without affecting coarse FAISS retrieval.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


@dataclass
class GeometricVerificationResult:
    """Detailed output of geometric verification between two images."""

    is_verified: bool
    num_matches: int
    num_inliers: int
    inlier_ratio: float
    spatial_dispersion: float
    geometric_score: float
    homography: Optional[np.ndarray] = None


class GeometricVerifier:
    """Self-contained geometric verification using local features and robust estimation."""

    def __init__(
        self,
        detector_type: str = "sift",
        max_features: int = 1000,
        ratio_threshold: float = 0.75,
        min_inliers: int = 12,
        reproj_threshold: float = 3.0,
    ):
        if not HAS_OPENCV:
            raise ImportError(
                "OpenCV is required for geometric verification. "
                "Install it with `pip install opencv-python-headless`."
            )

        self.detector_type = detector_type.lower()
        self.max_features = max_features
        self.ratio_threshold = ratio_threshold
        self.min_inliers = min_inliers
        self.reproj_threshold = reproj_threshold

        if self.detector_type == "sift":
            self.detector = cv2.SIFT_create(nfeatures=self.max_features)
            # FLANN parameters for SIFT (KDTree)
            index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif self.detector_type == "orb":
            self.detector = cv2.ORB_create(nfeatures=self.max_features)
            # BFMatcher with Hamming distance for binary descriptors
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}. Choose 'sift' or 'orb'.")

    def _to_grayscale_numpy(self, img_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """Convert various image inputs into a single-channel uint8 numpy array."""
        if isinstance(img_input, (str, Path)):
            path = Path(img_input)
            if not path.is_file():
                raise FileNotFoundError(f"Image not found at {path}")
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to decode image at {path}")
            return img

        if isinstance(img_input, Image.Image):
            rgb = np.array(img_input.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        if isinstance(img_input, np.ndarray):
            if img_input.ndim == 2:
                return img_input.astype(np.uint8)
            if img_input.ndim == 3:
                if img_input.shape[2] == 3:
                    return cv2.cvtColor(img_input.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                if img_input.shape[2] == 4:
                    return cv2.cvtColor(img_input.astype(np.uint8), cv2.COLOR_RGBA2GRAY)

        raise TypeError(f"Unsupported image type: {type(img_input)}")

    def extract_features(
        self, img_input: Union[str, Path, Image.Image, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """Extract keypoints and descriptors from an image.

        Returns:
            keypoints: (N, 2) float32 coordinates [x, y]
            descriptors: (N, dim) descriptors array
            shape: (height, width)
        """
        gray = self._to_grayscale_numpy(img_input)
        h, w = gray.shape[:2]

        kpts, descs = self.detector.detectAndCompute(gray, None)
        if descs is None or len(kpts) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 128), dtype=np.float32), (h, w)

        kpt_coords = np.array([k.pt for k in kpts], dtype=np.float32)
        return kpt_coords, descs, (h, w)

    def _compute_spatial_dispersion(
        self, inlier_pts: np.ndarray, img_shape: Tuple[int, int], grid_size: int = 4
    ) -> float:
        """Compute the spatial entropy of inliers across a grid.

        Prevents false high scores from clustered matches on single repetitive textures.
        Returns a score in [0.0, 1.0].
        """
        if len(inlier_pts) == 0:
            return 0.0

        h, w = img_shape
        if h <= 0 or w <= 0:
            return 0.5

        # Discretize inlier coordinates into grid cells
        cell_x = np.clip((inlier_pts[:, 0] / w * grid_size).astype(int), 0, grid_size - 1)
        cell_y = np.clip((inlier_pts[:, 1] / h * grid_size).astype(int), 0, grid_size - 1)
        cell_indices = cell_y * grid_size + cell_x

        total_cells = grid_size * grid_size
        counts = np.bincount(cell_indices, minlength=total_cells)
        probs = counts / float(len(inlier_pts))

        # Shannon entropy
        non_zero_probs = probs[probs > 0]
        entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))
        max_entropy = np.log2(total_cells)

        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    def verify_pair(
        self,
        kpts1: np.ndarray,
        descs1: np.ndarray,
        shape1: Tuple[int, int],
        kpts2: np.ndarray,
        descs2: np.ndarray,
        shape2: Tuple[int, int],
    ) -> GeometricVerificationResult:
        """Verify geometric consistency between two pre-extracted sets of features."""
        if len(kpts1) < 4 or len(kpts2) < 4 or descs1 is None or descs2 is None:
            return GeometricVerificationResult(
                is_verified=False,
                num_matches=0,
                num_inliers=0,
                inlier_ratio=0.0,
                spatial_dispersion=0.0,
                geometric_score=0.0,
                homography=None,
            )

        # 1. Lowe's 2-NN ratio test
        try:
            knn_matches = self.matcher.knnMatch(descs1, descs2, k=2)
        except Exception:
            return GeometricVerificationResult(
                is_verified=False,
                num_matches=0,
                num_inliers=0,
                inlier_ratio=0.0,
                spatial_dispersion=0.0,
                geometric_score=0.0,
            )

        good_matches = []
        for m_tuple in knn_matches:
            if len(m_tuple) == 2:
                m, n = m_tuple
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)

        num_matches = len(good_matches)
        if num_matches < 4:
            return GeometricVerificationResult(
                is_verified=False,
                num_matches=num_matches,
                num_inliers=0,
                inlier_ratio=0.0,
                spatial_dispersion=0.0,
                geometric_score=0.0,
                homography=None,
            )

        # 2. Extract matched point coordinates
        src_pts = np.float32([kpts1[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpts2[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)

        # 3. Robust homography estimation via USAC_MAGSAC (fallback to RANSAC)
        method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        try:
            H, mask = cv2.findHomography(src_pts, dst_pts, method, self.reproj_threshold)
        except Exception:
            H, mask = None, None

        if H is None or mask is None:
            return GeometricVerificationResult(
                is_verified=False,
                num_matches=num_matches,
                num_inliers=0,
                inlier_ratio=0.0,
                spatial_dispersion=0.0,
                geometric_score=0.0,
                homography=None,
            )

        inliers_mask = mask.ravel().astype(bool)
        num_inliers = int(np.sum(inliers_mask))
        inlier_ratio = float(num_inliers / num_matches) if num_matches > 0 else 0.0

        # 4. Inlier spatial dispersion across the image grid
        inlier_coords = src_pts[inliers_mask].reshape(-1, 2)
        dispersion = self._compute_spatial_dispersion(inlier_coords, shape1)

        # 5. Composite geometric score: combines inlier count saturation, ratio, and spatial coverage
        inlier_term = np.tanh(num_inliers / 30.0)
        geom_score = float(inlier_term * (0.6 * inlier_ratio + 0.4 * dispersion))

        is_verified = (num_inliers >= self.min_inliers) and (inlier_ratio >= 0.15)

        return GeometricVerificationResult(
            is_verified=is_verified,
            num_matches=num_matches,
            num_inliers=num_inliers,
            inlier_ratio=round(inlier_ratio, 4),
            spatial_dispersion=round(dispersion, 4),
            geometric_score=round(geom_score, 4),
            homography=H,
        )

    def verify_images(
        self,
        img1: Union[str, Path, Image.Image, np.ndarray],
        img2: Union[str, Path, Image.Image, np.ndarray],
    ) -> GeometricVerificationResult:
        """Extract features and verify geometric consistency in one step."""
        kpts1, descs1, shape1 = self.extract_features(img1)
        kpts2, descs2, shape2 = self.extract_features(img2)
        return self.verify_pair(kpts1, descs1, shape1, kpts2, descs2, shape2)
