"""Pretrained CLIP Feature Extractor with L2 Normalization and Device Handling."""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import open_clip

from src.utils.logging import get_logger

logger = get_logger("clip_encoder")


def select_best_device(requested: str = "auto") -> torch.device:
    """Select appropriate device based on request and availability (CUDA > MPS > CPU)."""
    req = requested.strip().lower()
    if req in ("cuda", "gpu") and torch.cuda.is_available():
        return torch.device("cuda")
    if req == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if req == "cpu":
        return torch.device("cpu")

    # Auto selection
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CLIPEncoder:
    """OpenCLIP Image and Text Encoder producing L2-normalized unit embeddings."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "auto",
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = select_best_device(device)

        logger.info(
            "Loading OpenCLIP model '%s' (pretrained='%s') on device: %s",
            self.model_name,
            self.pretrained,
            self.device,
        )

        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained if self.pretrained not in ("none", "") else "",
                device=self.device,
            )
        except Exception as e:
            if self.pretrained and self.pretrained not in ("none", ""):
                logger.warning(
                    "Could not load pretrained weights '%s' (%s). "
                    "Falling back to architecture '%s' with uninitialized weights for offline execution.",
                    self.pretrained,
                    e,
                    self.model_name,
                )
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    self.model_name,
                    pretrained="",
                    device=self.device,
                )
            else:
                raise
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.eval()

        # Determine embedding dimension
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            dummy_emb = self.model.encode_image(dummy)
            self.embedding_dim = dummy_emb.shape[-1]

        logger.info("OpenCLIP model initialized successfully. Embedding dimension: %d", self.embedding_dim)

    def encode_pil_images(self, images: List[Image.Image]) -> np.ndarray:
        """Encode a batch of PIL images into L2-normalized float32 NumPy vectors."""
        if not images:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        tensors = [self.preprocess(img) for img in images]
        batch = torch.stack(tensors).to(self.device)

        with torch.inference_mode():
            features = self.model.encode_image(batch)
            # Normalize embeddings to unit L2 norm (so dot product equals cosine similarity)
            features = F.normalize(features, p=2, dim=-1)
            embeddings = features.detach().cpu().numpy().astype(np.float32)

        return embeddings

    def encode_image_paths(
        self,
        image_paths: List[Union[str, Path]],
    ) -> Tuple[np.ndarray, List[str], List[Tuple[str, str]]]:
        """Encode a list of image paths, skipping and reporting corrupt images.

        Returns:
            embeddings: (N_valid, dim) float32 array
            valid_paths: list of paths that were successfully read and embedded
            failed_records: list of (path, error_message)
        """
        valid_images: List[Image.Image] = []
        valid_paths: List[str] = []
        failed_records: List[Tuple[str, str]] = []

        for p in image_paths:
            path_str = str(p)
            try:
                with Image.open(path_str) as img:
                    img_rgb = img.convert("RGB")
                    valid_images.append(img_rgb)
                    valid_paths.append(path_str)
            except Exception as e:
                failed_records.append((path_str, str(e)))

        if not valid_images:
            return np.empty((0, self.embedding_dim), dtype=np.float32), [], failed_records

        embeddings = self.encode_pil_images(valid_images)
        return embeddings, valid_paths, failed_records

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a text query string into a normalized 1D embedding vector."""
        tokens = self.tokenizer([text]).to(self.device)
        with torch.inference_mode():
            text_features = self.model.encode_text(tokens)
            text_features = F.normalize(text_features, p=2, dim=-1)
            emb = text_features.detach().cpu().numpy().astype(np.float32)[0]
        return emb
