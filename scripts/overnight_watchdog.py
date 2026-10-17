#!/usr/bin/env python3
"""Autonomous Cloud VM Watchdog & Auto-Healing Daemon for GLDv2.

Monitors the active GPU VM, verifies the extraction pipeline is progressing,
restarts terminated Spot instances automatically, and if the current zone runs
out of Spot GPU capacity, automatically provisions a replacement VM in an
alternative zone (e.g. asia-southeast1-b, asia-northeast1-a/b/c, us-central1-a),
bootstraps it, and resumes embedding without duplicate computation.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "outputs" / "watchdog.log"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Candidate zones with L4 GPUs
FALLBACK_ZONES = [
    "asia-northeast1-b",
    "asia-northeast1-c",
    "us-central1-a",
    "us-central1-b",
    "us-central1-c",
    "asia-southeast1-c",
    "asia-southeast1-b",
]

IMAGE_FAMILY = "ubuntu-pro-accel-2604-amd64-nvidia-595-v20261016"
IMAGE_PROJECT = "ubuntu-os-accelerator-images"


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [WATCHDOG] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_cmd(cmd: str, timeout: int = 60) -> tuple[int, str]:
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return res.returncode, (res.stdout + "\n" + res.stderr).strip()
    except Exception as e:
        return 1, str(e)


def get_active_vm() -> tuple[str, str, str]:
    """Find any running or designated GLDv2 instance."""
    code, out = run_cmd("gcloud compute instances list --format='csv[no-heading](name,zone,status)'")
    if code != 0 or not out:
        return "", "", ""
    running_instances = []
    other_instances = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) >= 3 and parts[0].startswith("instance-"):
            if parts[2] == "RUNNING":
                running_instances.append((parts[0], parts[1], parts[2]))
            else:
                other_instances.append((parts[0], parts[1], parts[2]))
    if running_instances:
        # Sort to pick the newest by name timestamp
        running_instances.sort(key=lambda x: x[0], reverse=True)
        return running_instances[0]
    if other_instances:
        other_instances.sort(key=lambda x: x[0], reverse=True)
        return other_instances[0]
    return "", "", ""


def ensure_pipeline_running(vm_name: str, zone: str) -> bool:
    """Check if ingestion pipeline is active on the VM; if not, launch it."""
    code, out = run_cmd(
        f"gcloud compute ssh {vm_name} --zone={zone} --quiet --command='pgrep -f stream_all_chunks.py || pgrep -f ingest_splits.py'",
        timeout=25,
    )

    if code == 0 and out.strip():
        log(f"Pipeline is actively running on {vm_name} ({zone}).")
        return True

    log(f"Pipeline NOT running on {vm_name}. Starting it under tmux...")
    launch_cmd = (
        f"gcloud compute ssh {vm_name} --zone={zone} --quiet --command=\""
        f"tmux new-session -d -s gldv2_splits 'cd ~/visual-search-engine-gldv2 && "
        f".venv/bin/python scripts/ingest_splits.py --splits index test --batch-size 256 --num-workers 8 --device cuda'\""
    )
    code, out = run_cmd(launch_cmd, timeout=30)
    time.sleep(3)
    return True


def start_vm(vm_name: str, zone: str) -> bool:
    log(f"Attempting to restart {vm_name} in {zone}...")
    code, out = run_cmd(f"gcloud compute instances start {vm_name} --zone={zone}", timeout=90)
    if code == 0:
        log(f"Successfully restarted {vm_name}.")
        time.sleep(20)  # wait for sshd
        return True
    log(f"Restart failed for {vm_name}: {out}")
    return False


def sync_shards_to_mac(vm_name: str, zone: str):
    """Sync newest shards to Mac."""
    code, _ = run_cmd(
        f"gcloud compute scp --recurse --zone={zone} {vm_name}:visual-search-engine-gldv2/outputs/embeddings {OUTPUT_DIR}",
        timeout=180,
    )
    if code == 0:
        log(f"Synced latest shards from {vm_name} to local Mac.")


def provision_new_vm() -> tuple[str, str]:
    """Provision a new L4 Spot VM in the next available zone if current is unschedulable."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    new_name = f"instance-{timestamp}"

    for zone in FALLBACK_ZONES:
        log(f"Attempting to provision replacement VM {new_name} in {zone}...")
        create_cmd = (
            f"gcloud compute instances create {new_name} "
            f"--zone={zone} "
            f"--machine-type=g2-standard-16 "
            f"--provisioning-model=SPOT "
            f"--instance-termination-action=STOP "
            f"--image={IMAGE_FAMILY} "
            f"--image-project={IMAGE_PROJECT} "
            f"--boot-disk-size=200GB "
            f"--boot-disk-type=hyperdisk-balanced"
        )
        code, out = run_cmd(create_cmd, timeout=120)
        if code == 0:
            log(f"Successfully provisioned replacement VM {new_name} in {zone}!")
            time.sleep(30)

            # Bootstrap VM
            log(f"Bootstrapping environment on {new_name}...")
            setup_cmd = (
                f"gcloud compute ssh {new_name} --zone={zone} --quiet --command=\""
                f"sudo apt-get update && sudo apt-get install -y git aria2 tmux && "
                f"curl -LsSf https://astral.sh/uv/install.sh | sh && "
                f"~/.local/bin/uv python install 3.11 && "
                f"git clone https://github.com/yashgarg195/visual-search-engine-gldv2.git && "
                f"cd visual-search-engine-gldv2 && "
                f"~/.local/bin/uv venv .venv --python 3.11 && "
                f"~/.local/bin/uv pip install --python .venv torch torchvision --index-url https://download.pytorch.org/whl/cu121 && "
                f"~/.local/bin/uv pip install --python .venv faiss-gpu-cu12 open_clip_torch pandas pyarrow pillow tqdm pyyaml pytest && "
                f"mkdir -p outputs/embeddings data/processed\""
            )
            run_cmd(setup_cmd, timeout=600)

            # Upload pre-built metadata tables (index, test)
            run_cmd(f"gcloud compute ssh {new_name} --zone={zone} --quiet --command='mkdir -p ~/visual-search-engine-gldv2/data/processed'", timeout=30)
            for meta_name in ["metadata_index.parquet", "metadata_test.parquet"]:
                local_meta = PROJECT_ROOT / "data" / "processed" / meta_name
                if local_meta.is_file():
                    log(f"Uploading pre-built {meta_name} to {new_name}...")
                    run_cmd(f"gcloud compute scp --zone={zone} {local_meta} {new_name}:~/visual-search-engine-gldv2/data/processed/", timeout=120)

            # Upload local manifest so VM resumes from shard 410
            manifest_file = PROJECT_ROOT / "outputs" / "embeddings" / "embedding_manifest.json"
            if manifest_file.is_file():
                log(f"Uploading existing local manifest to {new_name}...")
                run_cmd(f"gcloud compute scp --zone={zone} {manifest_file} {new_name}:~/visual-search-engine-gldv2/outputs/embeddings/", timeout=30)

            ensure_pipeline_running(new_name, zone)
            return new_name, zone
        else:
            log(f"Zone {zone} rejected provisioning: {out}")

    log("CRITICAL: All fallback zones exhausted for Spot GPU provisioning.")
    return "", ""


def main():
    log("Autonomous overnight watchdog started.")
    while True:
        try:
            name, zone, status = get_active_vm()
            if not name:
                log("No active VM found. Attempting to provision replacement...")
                name, zone = provision_new_vm()
            elif status == "TERMINATED":
                log(f"Detected VM {name} ({zone}) is TERMINATED.")
                if not start_vm(name, zone):
                    log(f"Could not restart {name}. Provisioning replacement VM in alternative zone...")
                    name, zone = provision_new_vm()
                else:
                    ensure_pipeline_running(name, zone)
            elif status == "RUNNING":
                ensure_pipeline_running(name, zone)
                sync_shards_to_mac(name, zone)

        except Exception as e:
            log(f"Watchdog iteration error: {e}")

        time.sleep(120)


if __name__ == "__main__":
    main()
