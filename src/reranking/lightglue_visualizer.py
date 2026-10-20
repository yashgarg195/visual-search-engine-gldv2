"""Visual Keypoint Matcher & Feature Line Renderer using SuperPoint + LightGlue.

Renders side-by-side verification images with matched feature lines connecting corresponding
architectural keypoints between the user's query image and canonical reference images.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from src.reranking.geometric_verifier import GeometricVerificationResult
from src.reranking.lightglue_verifier import LightGlueVerifier
from src.utils.logging import get_logger

logger = get_logger("lightglue_visualizer")

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    import torch
    from lightglue.utils import rbd
    HAS_LIGHTGLUE = True
except ImportError:
    HAS_LIGHTGLUE = False


class LightGlueVisualizer:
    """Computes keypoint correspondences and renders side-by-side feature line visualizations."""

    def __init__(
        self,
        verifier: Optional[LightGlueVerifier] = None,
        output_dir: Optional[Path] = None,
    ):
        self.verifier = verifier or LightGlueVerifier(max_num_keypoints=1024, max_image_size=1024, depth_confidence=-1.0)
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def match_and_visualize(
        self,
        query_image_path: Union[str, Path],
        reference_image_path: Union[str, Path],
        max_lines_to_draw: int = 60,
        precomputed_query_features: Optional[Tuple[np.ndarray, Dict[str, Any], Tuple[int, int]]] = None,
    ) -> Dict[str, Any]:
        """Run LightGlue, compute homography inliers, and render a side-by-side match image."""
        q_path = Path(query_image_path)
        r_path = Path(reference_image_path)

        if not q_path.is_file() or not r_path.is_file():
            return {
                "is_verified": False,
                "inliers": 0,
                "confidence": 0.0,
                "visualization_path": None,
                "visualization_url": None,
            }

        # 1. Load images using OpenCV
        img0_bgr = cv2.imread(str(q_path))
        img1_bgr = cv2.imread(str(r_path))

        if img0_bgr is None or img1_bgr is None:
            return {
                "is_verified": False,
                "inliers": 0,
                "confidence": 0.0,
                "visualization_path": None,
                "visualization_url": None,
            }

        # 2. Extract SuperPoint features (reusing precomputed query features if supplied)
        if precomputed_query_features is not None:
            kpts0, feats0, shape0 = precomputed_query_features
        else:
            kpts0, feats0, shape0 = self.verifier.extract_features(q_path)
        kpts1, feats1, shape1 = self.verifier.extract_features(r_path)

        # 3. Match with LightGlue
        f0 = {k: v.to(self.verifier.device) if isinstance(v, torch.Tensor) else v for k, v in feats0.items()}
        f1 = {k: v.to(self.verifier.device) if isinstance(v, torch.Tensor) else v for k, v in feats1.items()}

        with torch.no_grad():
            matches_res = self.verifier.matcher({"image0": f0, "image1": f1})
            matches_res = rbd(matches_res)

        matches = matches_res.get("matches")
        if matches is None or len(matches) < 4:
            return {
                "is_verified": False,
                "inliers": 0,
                "confidence": 0.0,
                "visualization_path": None,
                "visualization_url": None,
            }

        matches_np = matches.cpu().numpy()
        m_kpts0 = np.float32(kpts0[matches_np[:, 0]])
        m_kpts1 = np.float32(kpts1[matches_np[:, 1]])

        # 4. Homography with USAC_MAGSAC
        method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        try:
            H, mask = cv2.findHomography(m_kpts0, m_kpts1, method, self.verifier.reproj_threshold)
        except Exception:
            H, mask = None, None

        if H is None or mask is None:
            num_inliers = 0
            inlier_mask = np.zeros(len(matches_np), dtype=bool)
        else:
            inlier_mask = mask.ravel().astype(bool)
            num_inliers = int(np.sum(inlier_mask))

        # Check geometric sanity of homography
        is_h_valid = self.verifier.is_valid_homography(H, shape0, shape1)

        # Extract neural attention match confidence scores
        scores_res = matches_res.get("scores")
        if scores_res is not None and len(scores_res) == len(matches_np):
            scores_np = scores_res.cpu().numpy()
            if num_inliers > 0:
                inlier_scores = scores_np[inlier_mask]
                mean_inlier_conf = float(np.mean(inlier_scores)) if len(inlier_scores) > 0 else 0.5
            else:
                mean_inlier_conf = float(np.mean(scores_np)) if len(scores_np) > 0 else 0.0
        else:
            mean_inlier_conf = 0.5

        inlier_ratio = float(num_inliers / max(1, len(matches_np)))

        # Continuous, calibrated confidence score
        if num_inliers >= 4:
            inlier_factor = min(1.0, num_inliers / 12.0)
            confidence = float(min(1.0, max(0.20, inlier_factor * (0.40 + 0.60 * mean_inlier_conf))))
            pts0_to_draw = m_kpts0[inlier_mask]
            pts1_to_draw = m_kpts1[inlier_mask]
        elif len(matches_np) > 0:
            confidence = float(min(0.35, max(0.10, (len(matches_np) / 20.0) * mean_inlier_conf)))
            pts0_to_draw = m_kpts0
            pts1_to_draw = m_kpts1
        else:
            confidence = 0.0
            pts0_to_draw = np.empty((0, 2), dtype=np.float32)
            pts1_to_draw = np.empty((0, 2), dtype=np.float32)

        # 5. Render Side-by-Side Visualization Image
        vis_filename = self._generate_vis_filename(q_path, r_path)
        out_vis_path = self.output_dir / vis_filename

        self._render_feature_lines(
            img0_bgr,
            img1_bgr,
            pts0_to_draw,
            pts1_to_draw,
            num_inliers,
            confidence,
            out_vis_path,
            max_lines=max_lines_to_draw,
        )

        return {
            "is_verified": is_h_valid and (num_inliers >= self.verifier.min_inliers) and (confidence >= 0.20),
            "inliers": num_inliers,
            "confidence": round(confidence, 3),
            "visualization_path": str(out_vis_path),
            "visualization_url": f"/visualizations/{vis_filename}",
        }

    def _generate_vis_filename(self, path0: Path, path1: Path) -> str:
        key = f"{path0.name}_{path1.name}_{path0.stat().st_mtime}_{path1.stat().st_mtime}"
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return f"match_{h}.jpg"

    def _render_feature_lines(
        self,
        img0: np.ndarray,
        img1: np.ndarray,
        pts0: np.ndarray,
        pts1: np.ndarray,
        inliers: int,
        confidence: float,
        out_path: Path,
        max_lines: int = 60,
    ) -> None:
        """Draw matched keypoints and connecting lines between two images side by side."""
        # Target unified height
        target_h = 540
        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]

        scale0 = target_h / float(h0)
        scale1 = target_h / float(h1)

        new_w0 = int(round(w0 * scale0))
        new_w1 = int(round(w1 * scale1))

        resized0 = cv2.resize(img0, (new_w0, target_h), interpolation=cv2.INTER_AREA)
        resized1 = cv2.resize(img1, (new_w1, target_h), interpolation=cv2.INTER_AREA)

        # Scale keypoints to match resized images
        scaled_pts0 = pts0 * scale0
        scaled_pts1 = pts1 * scale1

        # Create side-by-side canvas with header banner (target_h + 60px banner)
        banner_h = 50
        total_w = new_w0 + new_w1
        canvas = np.zeros((target_h + banner_h, total_w, 3), dtype=np.uint8)

        # Place header banner background
        canvas[:banner_h, :] = (26, 26, 30)  # Dark slate

        # Place images below banner
        canvas[banner_h:, :new_w0] = resized0
        canvas[banner_h:, new_w0:] = resized1

        # Draw header text
        color_badge = (72, 201, 107) if inliers >= 20 else (80, 160, 240)
        header_text = f"SuperPoint + LightGlue: {inliers} Architectural Inliers | Confidence: {int(confidence * 100)}%"
        cv2.putText(canvas, header_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        # Sub-labels for Query Image vs Canonical Reference
        cv2.putText(canvas, "YOUR QUERY IMAGE", (20, banner_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "CANONICAL REFERENCE", (new_w0 + 20, banner_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # Subsample lines if there are too many to prevent visual clutter
        total_pts = len(scaled_pts0)
        if total_pts > max_lines:
            step = total_pts / float(max_lines)
            indices = [int(i * step) for i in range(max_lines)]
        else:
            indices = list(range(total_pts))

        # Color palette for correspondence lines
        np.random.seed(42)
        colors = [
            (0, 255, 128),   # Spring green
            (0, 220, 255),   # Cyan
            (255, 180, 50),   # Sky blue
            (50, 100, 255),   # Coral
            (220, 80, 255),   # Magenta
            (100, 255, 220),  # Mint
        ]

        # Draw matching lines and circles
        for idx in indices:
            p0 = scaled_pts0[idx]
            p1 = scaled_pts1[idx]

            x0 = int(round(p0[0]))
            y0 = int(round(p0[1])) + banner_h

            x1 = int(round(p1[0])) + new_w0
            y1 = int(round(p1[1])) + banner_h

            c = colors[idx % len(colors)]

            # Draw circle on keypoints
            cv2.circle(canvas, (x0, y0), 4, c, -1, cv2.LINE_AA)
            cv2.circle(canvas, (x0, y0), 5, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.circle(canvas, (x1, y1), 4, c, -1, cv2.LINE_AA)
            cv2.circle(canvas, (x1, y1), 5, (0, 0, 0), 1, cv2.LINE_AA)

            # Draw smooth line connecting them
            cv2.line(canvas, (x0, y0), (x1, y1), c, 1, cv2.LINE_AA)

        # Draw separating divider
        cv2.line(canvas, (new_w0, banner_h), (new_w0, target_h + banner_h), (255, 255, 255), 2, cv2.LINE_AA)

        # Save to disk
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        logger.info("Saved feature match visualization with %d inliers to %s", inliers, out_path)
