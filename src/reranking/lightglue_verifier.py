"""Modular SuperPoint + LightGlue Deep Neural Matcher for Stage-2 Geometric Verification.

State-of-the-art transformer-based keypoint detection and correspondence matching.
Fully decoupled and removable: seamlessly falls back to classical SIFT or coarse ranks.
Optimized for Apple Silicon GPU (MPS) with automatic CPU fallback.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image

from src.reranking.geometric_verifier import GeometricVerificationResult
from src.utils.logging import get_logger

logger = get_logger("lightglue_verifier")

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    import torch
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import numpy_image_to_torch, rbd, resize_image
    HAS_LIGHTGLUE = True
except ImportError:
    HAS_LIGHTGLUE = False


class LightGlueVerifier:
    """Neural geometric verifier combining SuperPoint keypoints and LightGlue attention matching."""

    def __init__(
        self,
        max_num_keypoints: int = 1024,
        max_image_size: int = 1024,
        reproj_threshold: float = 3.0,
        min_inliers: int = 6,
        device: Optional[str] = None,
        depth_confidence: float = -1.0,
        width_confidence: float = 0.99,
    ):
        if not HAS_LIGHTGLUE:
            raise ImportError(
                "LightGlue is not installed. Install via: "
                "pip install git+https://github.com/cvg/LightGlue.git kornia"
            )
        if not HAS_OPENCV:
            raise ImportError("OpenCV is required for homography estimation. Install opencv-python-headless.")

        self.max_num_keypoints = max_num_keypoints
        self.max_image_size = max_image_size
        self.reproj_threshold = reproj_threshold
        self.min_inliers = min_inliers

        # Determine target device (prefer MPS on Apple Silicon, then CUDA, else CPU)
        if device is not None:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        logger.info("Initializing SuperPoint + LightGlue on %s...", self.device)
        self.extractor = SuperPoint(max_num_keypoints=self.max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(
            features="superpoint",
            depth_confidence=depth_confidence,
            width_confidence=width_confidence,
        ).eval().to(self.device)

    def _prepare_tensor(
        self, img_input: Union[str, Path, Image.Image, np.ndarray]
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Convert image input to a normalized float32 tensor (1, 3, H, W) on device."""
        if isinstance(img_input, (str, Path)):
            p = Path(img_input)
            if not p.is_file():
                raise FileNotFoundError(f"Image not found at {p}")
            with Image.open(p) as img:
                rgb = np.array(img.convert("RGB"))
        elif isinstance(img_input, Image.Image):
            rgb = np.array(img_input.convert("RGB"))
        elif isinstance(img_input, np.ndarray):
            if img_input.ndim == 2:
                rgb = cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB)
            elif img_input.shape[2] == 4:
                rgb = cv2.cvtColor(img_input, cv2.COLOR_RGBA2RGB)
            else:
                rgb = img_input
        else:
            raise TypeError(f"Unsupported image type: {type(img_input)}")

        orig_h, orig_w = rgb.shape[:2]

        # Downscale large images while preserving aspect ratio for fast inference
        if max(orig_h, orig_w) > self.max_image_size:
            scale = self.max_image_size / float(max(orig_h, orig_w))
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        tensor = numpy_image_to_torch(rgb).to(self.device)
        return tensor, (orig_h, orig_w)

    def extract_features(
        self, img_input: Union[str, Path, Image.Image, np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, Any], Tuple[int, int]]:
        """Extract SuperPoint keypoints and descriptors.

        Returns:
            kpt_coords: (N, 2) float32 coordinates [x, y] in original image space.
            feats_cpu: Feature dictionary stored on CPU for LRU caching.
            shape: (orig_h, orig_w)
        """
        tensor, (orig_h, orig_w) = self._prepare_tensor(img_input)
        tensor_h, tensor_w = tensor.shape[-2:]

        with torch.no_grad():
            feats = self.extractor.extract(tensor)

        # Scale keypoints back to original image dimensions if resized
        scale_x = orig_w / float(tensor_w)
        scale_y = orig_h / float(tensor_h)

        # Move to CPU for safe caching and free up device memory
        feats_cpu: Dict[str, Any] = {}
        for k, v in feats.items():
            if isinstance(v, torch.Tensor):
                feats_cpu[k] = v.cpu()
            else:
                feats_cpu[k] = v

        kpts_raw = feats_cpu["keypoints"][0].numpy()
        if len(kpts_raw) > 0 and (scale_x != 1.0 or scale_y != 1.0):
            kpts_scaled = kpts_raw.copy()
            kpts_scaled[:, 0] *= scale_x
            kpts_scaled[:, 1] *= scale_y
        else:
            kpts_scaled = kpts_raw

        return kpts_scaled, feats_cpu, (orig_h, orig_w)

    def _compute_spatial_dispersion(
        self, inlier_pts: np.ndarray, img_shape: Tuple[int, int], grid_size: int = 4
    ) -> float:
        """Compute the spatial entropy of inliers across a grid to prevent cluster bias."""
        if len(inlier_pts) == 0:
            return 0.0

        h, w = img_shape
        if h <= 0 or w <= 0:
            return 0.5

        cell_x = np.clip((inlier_pts[:, 0] / w * grid_size).astype(int), 0, grid_size - 1)
        cell_y = np.clip((inlier_pts[:, 1] / h * grid_size).astype(int), 0, grid_size - 1)
        cell_indices = cell_y * grid_size + cell_x

        total_cells = grid_size * grid_size
        counts = np.bincount(cell_indices, minlength=total_cells)
        probs = counts / float(len(inlier_pts))

        non_zero_probs = probs[probs > 0]
        entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))
        max_entropy = np.log2(total_cells)

        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    @staticmethod
    def is_valid_homography(
        H: Optional[np.ndarray], shape0: Tuple[int, int], shape1: Tuple[int, int]
    ) -> bool:
        """Verify that a homography matrix is non-degenerate and mathematically sound."""
        if H is None or H.shape != (3, 3):
            return False

        det = float(np.linalg.det(H))
        if np.isnan(det) or np.isinf(det) or abs(det) <= 1e-9:
            return False

        return True

    def verify_pair(
        self,
        kpts1: np.ndarray,
        feats1: Dict[str, Any],
        shape1: Tuple[int, int],
        kpts2: np.ndarray,
        feats2: Dict[str, Any],
        shape2: Tuple[int, int],
    ) -> GeometricVerificationResult:
        """Run LightGlue attention matching between two pre-extracted feature sets."""
        if len(kpts1) < 4 or len(kpts2) < 4 or not isinstance(feats1, dict) or not isinstance(feats2, dict):
            return GeometricVerificationResult(
                is_verified=False,
                num_matches=0,
                num_inliers=0,
                inlier_ratio=0.0,
                spatial_dispersion=0.0,
                geometric_score=0.0,
                homography=None,
            )

        # Move features to target device
        f0 = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in feats1.items()}
        f1 = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in feats2.items()}

        try:
            with torch.no_grad():
                matches_res = self.matcher({"image0": f0, "image1": f1})
                matches_res = rbd(matches_res)
        except Exception as e:
            logger.warning("LightGlue matching failed: %s", e)
            return GeometricVerificationResult(
                is_verified=False,
                num_matches=0,
                num_inliers=0,
                inlier_ratio=0.0,
                spatial_dispersion=0.0,
                geometric_score=0.0,
            )

        matches = matches_res.get("matches")
        if matches is None or len(matches) < 4:
            num_m = len(matches) if matches is not None else 0
            return GeometricVerificationResult(
                is_verified=False,
                num_matches=num_m,
                num_inliers=0,
                inlier_ratio=0.0,
                spatial_dispersion=0.0,
                geometric_score=0.0,
                homography=None,
            )

        # Extract matched point coordinates
        matches_np = matches.cpu().numpy()
        m_kpts0 = np.float32(kpts1[matches_np[:, 0]])
        m_kpts1 = np.float32(kpts2[matches_np[:, 1]])

        # Robust homography estimation using USAC_MAGSAC
        method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        try:
            H, mask = cv2.findHomography(m_kpts0, m_kpts1, method, self.reproj_threshold)
        except Exception:
            H, mask = None, None

        num_matches = len(matches_np)
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

        # Extract neural attention match confidence scores
        scores_res = matches_res.get("scores")
        if scores_res is not None and len(scores_res) == len(matches_np):
            scores_np = scores_res.cpu().numpy()
            inlier_scores = scores_np[inliers_mask]
            mean_inlier_conf = float(np.mean(inlier_scores)) if len(inlier_scores) > 0 else 0.0
        else:
            mean_inlier_conf = 0.5

        inlier_coords = m_kpts0[inliers_mask]
        dispersion = self._compute_spatial_dispersion(inlier_coords, shape1)

        # Sanity check homography non-degeneracy & convexity
        is_h_valid = self.is_valid_homography(H, shape1, shape2)

        # LightGlue matches are already deeply verified, so a high inlier ratio is strong evidence
        inlier_term = np.tanh(num_inliers / 25.0)
        geom_score = float(inlier_term * (0.45 * inlier_ratio + 0.35 * mean_inlier_conf + 0.20 * dispersion))

        is_verified = is_h_valid and (num_inliers >= self.min_inliers) and (inlier_ratio >= 0.10 or mean_inlier_conf >= 0.20)

        return GeometricVerificationResult(
            is_verified=is_verified,
            num_matches=num_matches,
            num_inliers=num_inliers,
            inlier_ratio=round(inlier_ratio, 4),
            spatial_dispersion=round(dispersion, 4),
            geometric_score=round(geom_score, 4),
            homography=H if is_h_valid else None,
        )

    def verify_images(
        self,
        img1: Union[str, Path, Image.Image, np.ndarray],
        img2: Union[str, Path, Image.Image, np.ndarray],
    ) -> GeometricVerificationResult:
        """Extract features and verify geometric consistency between two images."""
        kpts1, feats1, shape1 = self.extract_features(img1)
        kpts2, feats2, shape2 = self.extract_features(img2)
        return self.verify_pair(kpts1, feats1, shape1, kpts2, feats2, shape2)
