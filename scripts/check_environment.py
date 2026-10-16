#!/usr/bin/env python3
"""Environment, Hardware, and Dependency Checker for Visual Landmark Search Pipeline.

Inspects system hardware (CPU, RAM, GPU/CUDA/MPS), disk space at data locations,
library versions (PyTorch, OpenCLIP, FAISS, etc.), and GLDv2 dataset directory readiness.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.utils.hardware import check_system_hardware


def format_status(val: bool) -> str:
    return "AVAILABLE" if val else "NOT AVAILABLE"


def main():
    parser = argparse.ArgumentParser(
        description="Check execution environment, hardware resources, libraries, and dataset paths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", type=str, default=".env", help="Path to .env configuration file")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of human-readable text")
    args = parser.parse_args()

    cfg = get_config(args.env_file)

    report = check_system_hardware(
        data_paths=[cfg.data_root, cfg.image_dir, cfg.embedding_dir, cfg.index_dir],
        image_dir=cfg.image_dir,
        metadata_dir=cfg.metadata_dir,
        processed_dir=cfg.processed_dir,
    )

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("=" * 80)
    print(" SYSTEM ENVIRONMENT & HARDWARE AUDIT REPORT")
    print("=" * 80)

    # 1. System & Python
    sys_info = report["system"]
    print(f"\n[OS & Runtime]")
    print(f"  OS Platform        : {sys_info['os']} {sys_info['release']} ({sys_info['machine']})")
    print(f"  Python Version     : {sys_info['python_version']}")
    print(f"  Python Binary      : {sys_info['python_executable']}")

    # 2. CPU & RAM
    cpu_info = report["cpu"]
    ram_info = report["ram"]
    print(f"\n[Compute & Memory]")
    print(f"  CPU Cores          : {cpu_info['logical_cores']} logical, {cpu_info['physical_cores']} physical")
    if ram_info["total_gb"]:
        print(f"  RAM (System)       : {ram_info['available_gb']} GB available / {ram_info['total_gb']} GB total")
    else:
        print("  RAM                : Could not determine available RAM (install psutil for detailed stats)")

    # 3. GPU / Acceleration
    gpu_info = report["gpu"]
    print(f"\n[Hardware Acceleration]")
    print(f"  CUDA Available     : {gpu_info['cuda_available']}")
    if gpu_info["cuda_available"]:
        print(f"  CUDA Version       : {gpu_info['cuda_version']}")
        print(f"  GPU Count          : {gpu_info['device_count']}")
        for dev in gpu_info["devices"]:
            print(f"    - GPU {dev['id']}: {dev['name']} ({dev['vram_gb']} GB VRAM, CC {dev['compute_capability']})")
    else:
        print("  CUDA Status        : CUDA GPU not detected (CPU inference will be used).")

    if gpu_info["mps_available"]:
        print("  Apple MPS (Metal)  : Available (Supported for local development on Apple Silicon)")

    print(f"  Recommended Device : {gpu_info['recommended_device'].upper()}")
    for note in gpu_info["notes"]:
        print(f"    NOTE: {note}")

    # 4. Libraries
    print(f"\n[Software Dependencies]")
    pkgs = report["packages"]
    for pkg_name, ver in pkgs.items():
        status = f"v{ver}" if ver else "MISSING (Optional or not installed)"
        print(f"  {pkg_name:<18} : {status}")

    # 5. Disk Space
    print(f"\n[Storage & Disk Space]")
    for path_str, usage in report["disk"].items():
        if "error" in usage:
            print(f"  {path_str:<25} : Error inspecting ({usage['error']})")
        else:
            print(
                f"  {path_str:<25} : {usage['free_gb']} GB free / {usage['total_gb']} GB total ({usage['percent_used']}% used)"
            )

    # 6. Dataset & Metadata
    ds = report.get("dataset", {})
    print(f"\n[Dataset & Metadata Paths]")
    print(f"  Image Dir Exists   : {ds.get('image_dir_exists')} ({cfg.image_dir})")
    print(f"  Metadata Dir Exists: {ds.get('metadata_dir_exists')} ({cfg.metadata_dir})")
    print(f"  Processed Dir Exists: {ds.get('processed_dir_exists')} ({cfg.processed_dir})")

    meta_files = ds.get("metadata_files", {})
    print("  Official Metadata Files:")
    for key, item in meta_files.items():
        status_str = f"Found ({item['size_bytes']} bytes)" if item["exists"] else "Not Found"
        print(f"    - {Path(item['path']).name:<35}: {status_str}")

    print("\n" + "=" * 80)
    print(" SUMMARY RECOMMENDATION")
    print("=" * 80)
    all_core_ok = all(pkgs.get(p) for p in ["torch", "open_clip", "faiss", "numpy", "pandas", "PIL"])
    if all_core_ok:
        print(" [OK] Core dependencies (PyTorch, OpenCLIP, FAISS, NumPy, Pandas, Pillow) are ready.")
    else:
        print(" [WARN] Some core packages are missing. Install with 'pip install -r requirements.txt'")

    if gpu_info["cuda_available"]:
        print(" [OK] CUDA acceleration active. High-throughput GPU inference enabled.")
    elif gpu_info["mps_available"]:
        print(" [INFO] Running on macOS with Apple Silicon MPS acceleration.")
    else:
        print(" [INFO] Running in CPU mode. Use batch size 16-64 for inference.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
