"""Atomic file I/O operations and manifest management."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def atomic_save_json(data: Any, target_path: Path, indent: int = 2) -> None:
    """Save data to JSON atomically using a temporary file."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp.{os.getpid()}")

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, target_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def load_json(path: Path) -> Any:
    """Load JSON file safely."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_save_numpy(arr: np.ndarray, target_path: Path) -> None:
    """Save NumPy array atomically."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp.{os.getpid()}")

    try:
        with open(temp_path, "wb") as f:
            np.save(f, arr)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, target_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def atomic_save_parquet(df: pd.DataFrame, target_path: Path) -> None:
    """Save DataFrame to Parquet atomically."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp.{os.getpid()}")

    try:
        df.to_parquet(temp_path, engine="pyarrow", index=False)
        os.replace(temp_path, target_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
