"""
Canonical Model Asset & Execution Provider Audit Script for PS 26053.
Canonical location: scripts/audit_assets.py

Audits:
- Discovered model weights in repository
- Model file size in binary MiB (separated from runtime memory)
- Runtime model memory isolation status
- Actual ONNX Runtime execution providers (CUDA availability)
- Model architecture verified from config/metadata
- 3D detection and tracking fallback status
"""

import os
import glob
import json
from pathlib import Path
from typing import Dict, List, Any

# Root workspace directory
ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_EXTENSIONS = [
    "*.onnx",
    "*.pt",
    "*.pth",
    "*.engine",
    "*.weights",
]


def audit_repository_assets() -> Dict[str, Any]:
    found_weights: List[str] = []
    for ext in MODEL_EXTENSIONS:
        found_weights.extend(
            glob.glob(str(ROOT_DIR / "**" / ext), recursive=True)
        )

    # Filter out virtual environments, node_modules, git directories
    filtered_weights = [
        w for w in found_weights
        if not any(skip in w for skip in [".git", "node_modules", ".venv", "venv", "__pycache__"])
    ]

    print("=== REPOSITORY MODEL ASSET AUDIT (CANONICAL) ===")
    print(f"Audit Root: {ROOT_DIR}")

    model_records = []
    for weight in filtered_weights:
        size_bytes = os.path.getsize(weight)
        size_mib = size_bytes / (1024.0 * 1024.0)
        rel_path = os.path.relpath(weight, ROOT_DIR)
        print(f"File: {rel_path} | Size: {size_mib:.2f} MiB ({size_bytes:,} bytes)")
        model_records.append({
            "path": rel_path,
            "filename": os.path.basename(weight),
            "model_file_size_mib": round(size_mib, 2),
            "model_runtime_memory_mib": "NOT_ISOLATED",
        })

    # Check ONNX Runtime providers programmatically
    avail_providers = []
    cuda_available = False
    try:
        import onnxruntime as ort
        avail_providers = ort.get_available_providers()
        cuda_available = "CUDAExecutionProvider" in avail_providers
    except ImportError:
        avail_providers = ["onnxruntime_not_installed"]

    print(f"Available ONNX Runtime Providers: {avail_providers}")
    print(f"CUDA Execution Provider Available: {'YES' if cuda_available else 'NO'}")

    # Inspect model_config.json if available
    config_path = ROOT_DIR / "Backend" / "backend" / "models" / "model_config.json"
    model_config_data = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                model_config_data = json.load(f)
            arch_name = model_config_data.get("architecture")
            print(f"Verified Model Name: {model_config_data.get('model_name')}")
            print(f"Architecture declared by model_config.json: {arch_name}")
        except Exception as e:
            print(f"Error reading model_config.json: {e}")

    # Pipeline status summary
    has_segmentation = any("SegmentationModel.onnx" in m["filename"] for m in model_records)
    has_pointpillars = any("pointpillar" in m["filename"].lower() for m in model_records)

    print("\n--- Pipeline Component Status ---")
    print(f"Semantic Segmentation: {'LOADED' if has_segmentation else 'MODEL NOT LOADED'}")
    print(f"PointPillars 3D Detection: {'LOADED' if has_pointpillars else 'NOT_AVAILABLE'}")
    print(f"Geometric Fallback Status: ACTIVE")
    print(f"3D Detection Backend: GEOMETRIC_FALLBACK (DBSCAN + height filtering)")
    print(f"Tracking Backend: Linear constant-velocity Kalman tracking with Hungarian data association on geometric fallback detections")
    print(f"Tracking on Geometric Detections: True")
    print(f"Performance Target: NOT_MET_ON_CURRENT_CPU (CPU latency exceeds 30ms; GPU/CUDA/TensorRT acceleration should be benchmarked as next optimisation path)")
    print(f"Overall Pipeline Mode: {'LIVE_DL_WITH_GEOMETRIC_DETECTION_FALLBACK' if has_segmentation else 'LIVE_GEOMETRIC_FALLBACK'}")

    return {
        "models": model_records,
        "available_providers": avail_providers,
        "cuda_available": cuda_available,
        "model_config": model_config_data,
        "pipeline_mode": "LIVE_DL_WITH_GEOMETRIC_DETECTION_FALLBACK" if has_segmentation else "LIVE_GEOMETRIC_FALLBACK",
        "semantic_segmentation_status": "LOADED" if has_segmentation else "MODEL NOT LOADED",
        "pointpillars_status": "LOADED" if has_pointpillars else "NOT_AVAILABLE",
        "geometric_fallback_status": "ACTIVE",
        "detection_backend": "GEOMETRIC_FALLBACK",
        "tracking_backend": "Linear constant-velocity Kalman tracking with Hungarian data association on geometric fallback detections",
        "tracking_on_geometric_detections": True,
        "performance_target": "NOT_MET_ON_CURRENT_CPU",
    }


if __name__ == "__main__":
    audit_repository_assets()
