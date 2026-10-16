"""Pretrained Vision & Multimodal Encoders."""

from src.models.clip_encoder import CLIPEncoder, select_best_device

__all__ = ["CLIPEncoder", "select_best_device"]
