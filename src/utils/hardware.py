"""Hardware, Environment and Dependency Inspection Utility."""

import importlib
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_ram_info() -> Dict[str, Any]:
    """Retrieve total and available RAM across Linux and macOS."""
    info: Dict[str, Any] = {"total_gb": None, "available_gb": None, "source": "unknown"}
    try:
        import psutil

        mem = psutil.virtual_memory()
        info["total_gb"] = round(mem.total / (1024**3), 2)
        info["available_gb"] = round(mem.available / (1024**3), 2)
        info["source"] = "psutil"
        return info
    except ImportError:
        pass

    # Linux /proc/meminfo fallback
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        try:
            mem_data = {}
            with open(meminfo_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = parts[1].strip().split()[0]
                        mem_data[key] = int(val)
            if "MemTotal" in mem_data:
                info["total_gb"] = round(mem_data["MemTotal"] / (1024**2), 2)
            if "MemAvailable" in mem_data:
                info["available_gb"] = round(mem_data["MemAvailable"] / (1024**2), 2)
            elif "MemFree" in mem_data:
                info["available_gb"] = round(mem_data["MemFree"] / (1024**2), 2)
            info["source"] = "/proc/meminfo"
            return info
        except Exception:
            pass

    # macOS sysctl fallback
    if platform.system() == "Darwin":
        try:
            import subprocess

            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            total_bytes = int(out)
            info["total_gb"] = round(total_bytes / (1024**3), 2)
            info["source"] = "sysctl"
            return info
        except Exception:
            pass

    return info


def get_cpu_info() -> Dict[str, Any]:
    """Retrieve CPU core count."""
    logical_cores = os.cpu_count() or 1
    physical_cores = None
    try:
        import psutil

        physical_cores = psutil.cpu_count(logical=False)
    except ImportError:
        pass

    return {
        "logical_cores": logical_cores,
        "physical_cores": physical_cores or logical_cores,
        "architecture": platform.machine(),
        "processor": platform.processor(),
    }


def get_gpu_info() -> Dict[str, Any]:
    """Inspect CUDA GPU availability, device name, VRAM, and Apple MPS support."""
    info: Dict[str, Any] = {
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
        "devices": [],
        "mps_available": False,
        "recommended_device": "cpu",
        "notes": [],
    }

    try:
        import torch

        # Check CUDA
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_version"] = torch.version.cuda
            info["device_count"] = torch.cuda.device_count()
            info["recommended_device"] = "cuda"
            for i in range(info["device_count"]):
                props = torch.cuda.get_device_properties(i)
                total_vram_gb = round(props.total_memory / (1024**3), 2)
                info["devices"].append({
                    "id": i,
                    "name": props.name,
                    "vram_gb": total_vram_gb,
                    "compute_capability": f"{props.major}.{props.minor}",
                })
        else:
            info["notes"].append("No CUDA GPU detected or PyTorch was built without CUDA.")

        # Check Apple MPS (Metal Performance Shaders on Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["mps_available"] = True
            if not info["cuda_available"]:
                info["recommended_device"] = "mps"
                info["notes"].append("Apple Silicon MPS acceleration is available for local Mac testing.")

    except ImportError:
        info["notes"].append("PyTorch is not installed. Device acceleration cannot be verified.")

    return info


def get_package_versions() -> Dict[str, Optional[str]]:
    """Inspect versions of all critical and optional libraries."""
    packages = [
        "torch",
        "torchvision",
        "open_clip",
        "faiss",
        "numpy",
        "pandas",
        "PIL",
        "pyarrow",
        "tqdm",
    ]
    versions: Dict[str, Optional[str]] = {}
    for pkg in packages:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "unknown")
            versions[pkg] = ver
        except ImportError:
            versions[pkg] = None
    return versions


def get_disk_info(paths: List[Path]) -> Dict[str, Dict[str, Any]]:
    """Retrieve total and free disk space for specified paths."""
    disk_data: Dict[str, Dict[str, Any]] = {}
    for p in paths:
        # Resolve existing ancestor if path does not exist yet
        check_path = p
        while not check_path.exists() and check_path.parent != check_path:
            check_path = check_path.parent

        try:
            usage = shutil.disk_usage(str(check_path))
            disk_data[str(p)] = {
                "checked_at": str(check_path),
                "total_gb": round(usage.total / (1024**3), 2),
                "used_gb": round(usage.used / (1024**3), 2),
                "free_gb": round(usage.free / (1024**3), 2),
                "percent_used": round((usage.used / usage.total) * 100, 1),
            }
        except Exception as e:
            disk_data[str(p)] = {"error": str(e)}
    return disk_data


def check_dataset_paths(
    image_dir: Path,
    metadata_dir: Path,
    processed_dir: Path,
) -> Dict[str, Any]:
    """Verify presence of dataset directories and expected official CSV files."""
    official_files = {
        "train_metadata": metadata_dir / "train.csv",
        "train_category": metadata_dir / "train_label_to_category.csv",
        "train_hierarchical": metadata_dir / "train_label_to_hierarchical.csv",
        "index_images": metadata_dir / "index.csv",
        "index_landmark_map": metadata_dir / "index_image_to_landmark.csv",
        "index_category": metadata_dir / "index_label_to_category.csv",
        "index_hierarchical": metadata_dir / "index_label_to_hierarchical.csv",
    }

    files_status = {}
    for key, p in official_files.items():
        files_status[key] = {
            "path": str(p),
            "exists": p.is_file(),
            "size_bytes": p.stat().st_size if p.is_file() else None,
        }

    return {
        "image_dir_exists": image_dir.is_dir(),
        "metadata_dir_exists": metadata_dir.is_dir(),
        "processed_dir_exists": processed_dir.is_dir(),
        "metadata_files": files_status,
    }


def check_system_hardware(
    data_paths: Optional[List[Path]] = None,
    image_dir: Optional[Path] = None,
    metadata_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Comprehensive environment audit returning a serializable dictionary."""
    data_paths = data_paths or [Path(".")]
    report = {
        "system": {
            "os": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
        },
        "cpu": get_cpu_info(),
        "ram": get_ram_info(),
        "gpu": get_gpu_info(),
        "packages": get_package_versions(),
        "disk": get_disk_info(data_paths),
    }

    if image_dir and metadata_dir and processed_dir:
        report["dataset"] = check_dataset_paths(image_dir, metadata_dir, processed_dir)

    return report
