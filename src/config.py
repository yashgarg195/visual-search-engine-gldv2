"""Pipeline Configuration and Environment Handling."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def load_env_file(dotenv_path: Optional[str] = None) -> None:
    """Lightweight loader for .env files without requiring python-dotenv."""
    path = Path(dotenv_path) if dotenv_path else Path(".env")
    if not path.is_file():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val


@dataclass
class PipelineConfig:
    """Master configuration for dataset, model, embedding store, and FAISS index."""

    # Storage paths
    data_root: Path = field(default_factory=lambda: Path(os.getenv("DATA_ROOT", "./data")))
    image_dir: Path = field(default_factory=lambda: Path(os.getenv("IMAGE_DIR", "./data/raw/images")))
    metadata_dir: Path = field(default_factory=lambda: Path(os.getenv("METADATA_DIR", "./data/raw/metadata")))
    processed_dir: Path = field(default_factory=lambda: Path(os.getenv("PROCESSED_DIR", "./data/processed")))
    embedding_dir: Path = field(default_factory=lambda: Path(os.getenv("EMBEDDING_DIR", "./outputs/embeddings")))
    index_dir: Path = field(default_factory=lambda: Path(os.getenv("INDEX_DIR", "./outputs/index")))
    eval_dir: Path = field(default_factory=lambda: Path(os.getenv("EVAL_DIR", "./outputs/evaluation")))

    # Download settings
    auto_download_metadata: bool = field(
        default_factory=lambda: os.getenv("AUTO_DOWNLOAD_METADATA", "false").lower() in ("true", "1", "yes")
    )

    # Model settings
    clip_model_name: str = field(default_factory=lambda: os.getenv("CLIP_MODEL_NAME", "ViT-B-32"))
    clip_pretrained: str = field(default_factory=lambda: os.getenv("CLIP_PRETRAINED", "openai"))
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "512")))
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "auto"))
    batch_size: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "64")))
    num_workers: int = field(default_factory=lambda: int(os.getenv("NUM_WORKERS", "4")))

    # Embedding shards
    shard_size: int = field(default_factory=lambda: int(os.getenv("SHARD_SIZE", "10000")))

    # FAISS IVF-PQ Index settings
    faiss_nlist: int = field(default_factory=lambda: int(os.getenv("FAISS_NLIST", "1024")))
    faiss_nprobe: int = field(default_factory=lambda: int(os.getenv("FAISS_NPROBE", "64")))
    faiss_m: int = field(default_factory=lambda: int(os.getenv("FAISS_M", "64")))
    faiss_nbits: int = field(default_factory=lambda: int(os.getenv("FAISS_NBITS", "8")))
    faiss_train_sample_size: int = field(
        default_factory=lambda: int(os.getenv("FAISS_TRAIN_SAMPLE_SIZE", "50000"))
    )
    random_seed: int = field(default_factory=lambda: int(os.getenv("RANDOM_SEED", "42")))

    def __post_init__(self):
        # Convert all paths to Path objects if passed as strings
        self.data_root = Path(self.data_root)
        self.image_dir = Path(self.image_dir)
        self.metadata_dir = Path(self.metadata_dir)
        self.processed_dir = Path(self.processed_dir)
        self.embedding_dir = Path(self.embedding_dir)
        self.index_dir = Path(self.index_dir)
        self.eval_dir = Path(self.eval_dir)

    def ensure_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        for path in [
            self.data_root,
            self.image_dir,
            self.metadata_dir,
            self.processed_dir,
            self.embedding_dir,
            self.index_dir,
            self.eval_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def get_config(env_file: Optional[str] = None) -> PipelineConfig:
    """Load configuration from environment, with optional .env file."""
    load_env_file(env_file)
    return PipelineConfig()
