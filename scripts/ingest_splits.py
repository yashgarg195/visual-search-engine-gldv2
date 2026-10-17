#!/usr/bin/env python3
"""Unified Multi-Split Ingestion Runner for GLDv2.

Orchestrates streaming download, GPU vectorization, and atomic sharding
for the official GLDv2 `index` and `test` splits, followed by incremental
FAISS index appending.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import get_logger

logger = get_logger("ingest_splits")


def run_command(cmd: list[str]) -> int:
    """Run a subprocess command with real-time output streaming."""
    logger.info("Executing: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT),
    )
    if proc.stdout:
        for line in iter(proc.stdout.readline, ""):
            print(line, end="", flush=True)
    proc.wait()
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="Ingest GLDv2 Index and Test splits")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["index", "test"],
        choices=["train", "index", "test"],
        help="Splits to ingest sequentially",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device (cuda, mps, cpu)")
    parser.add_argument("--cleanup-images-after-embed", action="store_true", default=True, help="Clean up images after embedding")
    parser.add_argument("--no-cleanup", dest="cleanup_images_after_embed", action="store_false", help="Keep downloaded raw images")
    parser.add_argument("--append-to-index", action="store_true", default=True, help="Append new shards to FAISS index after completion")
    args = parser.parse_args()

    python_bin = sys.executable
    stream_script = PROJECT_ROOT / "scripts" / "stream_all_chunks.py"
    add_script = PROJECT_ROOT / "scripts" / "add_embeddings.py"

    max_chunks = {"train": 499, "index": 99, "test": 19}

    t0 = time.time()
    for split in args.splits:
        end_chunk = max_chunks[split]
        logger.info("=== STARTING INGESTION FOR SPLIT: %s (0 to %d chunks) ===", split.upper(), end_chunk)

        cmd = [
            python_bin,
            str(stream_script),
            "--split", split,
            "--start-chunk", "0",
            "--end-chunk", str(end_chunk),
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--device", args.device,
        ]
        if args.cleanup_images_after_embed:
            cmd.append("--cleanup-images-after-embed")

        ret = run_command(cmd)
        if ret != 0:
            logger.error("Failed during ingestion of split %s (exit code %d). Aborting.", split, ret)
            return ret

    if args.append_to_index:
        logger.info("=== APPENDING NEW SHARDS TO FAISS IVF-PQ INDEX ===")
        ret = run_command([python_bin, str(add_script), "--checkpoint-interval", "20"])
        if ret != 0:
            logger.error("Failed appending new shards to index (exit code %d).", ret)
            return ret

    elapsed = time.time() - t0
    logger.info("=== ALL REQUESTED SPLITS INGESTED SUCCESSFULLY IN %.1f MINUTES ===", elapsed / 60.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
