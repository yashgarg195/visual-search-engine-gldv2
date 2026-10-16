"""GLDv2 CLIP + FAISS IVF-PQ Visual Landmark Search Pipeline."""

import os

# Prevent OpenMP multiple runtime conflict on macOS / Linux when PyTorch and FAISS coexist
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TQDM_MONITOR_INTERVAL", "0")

# Critical for macOS arm64: initialize torch runtime before faiss to prevent OpenMP segfault
try:
    import torch
except ImportError:
    pass

__version__ = "0.1.0"
