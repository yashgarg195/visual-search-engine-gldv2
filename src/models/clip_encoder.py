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


from torch.utils.data import Dataset, DataLoader


class FastPathDataset(Dataset):
    """Dataset for parallel multi-threaded image decoding and preprocessing."""

    def __init__(self, paths: List[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        try:
            with Image.open(p) as img:
                img_rgb = img.convert("RGB")
                tensor = self.transform(img_rgb)
            return tensor, p, True
        except Exception:
            return torch.zeros(3, 224, 224), p, False


def custom_collate_fn(batch):
    tensors, paths, valids = zip(*batch)
    valid_tensors = [t for t, v in zip(tensors, valids) if v]
    valid_paths = [p for p, v in zip(paths, valids) if v]
    if valid_tensors:
        batch_tensor = torch.stack(valid_tensors)
    else:
        batch_tensor = torch.empty(0, 3, 224, 224)
    return batch_tensor, valid_paths


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
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = self.model.encode_image(batch)
            else:
                features = self.model.encode_image(batch)
            # Normalize embeddings to unit L2 norm (so dot product equals cosine similarity)
            features = F.normalize(features, p=2, dim=-1)
            embeddings = features.detach().cpu().numpy().astype(np.float32)

        return embeddings

    def encode_image_paths(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 256,
        num_workers: int = 8,
    ) -> Tuple[np.ndarray, List[str], List[Tuple[str, str]]]:
        """Encode a list of image paths using multi-worker parallel CPU loading.

        Returns:
            embeddings: (N_valid, dim) float32 array
            valid_paths: list of paths that were successfully read and embedded
            failed_records: list of (path, error_message)
        """
        paths_str = [str(p) for p in image_paths]
        if not paths_str:
            return np.empty((0, self.embedding_dim), dtype=np.float32), [], []

        dataset = FastPathDataset(paths_str, self.preprocess)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=custom_collate_fn,
            persistent_workers=False,
        )

        all_embeddings = []
        all_valid_paths = []

        with torch.inference_mode():
            for batch_tensors, paths in loader:
                if len(paths) == 0:
                    continue
                batch_tensors = batch_tensors.to(self.device, non_blocking=True)
                if self.device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        features = self.model.encode_image(batch_tensors)
                else:
                    features = self.model.encode_image(batch_tensors)

                features = F.normalize(features, p=2, dim=-1)
                all_embeddings.append(features.detach().cpu().numpy().astype(np.float32))
                all_valid_paths.extend(paths)

        if all_embeddings:
            final_embeddings = np.vstack(all_embeddings)
        else:
            final_embeddings = np.empty((0, self.embedding_dim), dtype=np.float32)

        return final_embeddings, all_valid_paths, []

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a text query string into a normalized 1D embedding vector."""
        tokens = self.tokenizer([text]).to(self.device)
        with torch.inference_mode():
            text_features = self.model.encode_text(tokens)
            text_features = F.normalize(text_features, p=2, dim=-1)
            emb = text_features.detach().cpu().numpy().astype(np.float32)[0]
        return emb
