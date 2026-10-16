"""PyTest Configuration and Runtime Safeguards."""

import os

# Set before any numerical/deep learning libraries are imported
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TQDM_MONITOR_INTERVAL"] = "0"

# Import torch before faiss to prevent OpenMP runtime collision on macOS
import torch
