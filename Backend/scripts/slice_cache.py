import os
from pathlib import Path
import numpy as np


def downsample_and_write(src_path: Path, dest_path: Path, stride: int = 2):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.fromfile(str(src_path), dtype=np.float32).reshape(-1, 4)
    # Stride of 2 halves points and payload size (~800KB/file)
    downsampled = raw[::stride, :]
    downsampled.astype(np.float32).tofile(str(dest_path))


def run_slice():
    # Detect root vs Backend execution
    base_candidates = [
        Path("Backend/data/kaggle_cache/data"),
        Path("data/kaggle_cache/data"),
        Path(__file__).resolve().parent.parent / "data" / "kaggle_cache" / "data",
    ]
    cache_base = next((p for p in base_candidates if p.exists()), None)
    if not cache_base:
        raise FileNotFoundError("Could not find kaggle_cache/data directory.")

    static_src_bins = sorted(list((cache_base / "kitti_clean").rglob("*.bin")))[:50]
    dynamic_src_bins = sorted(list((cache_base / "kitti_dynamic").rglob("*.bin")))[:70]

    # Destination directories
    if Path("Backend/data").exists():
        static_dest = Path("Backend/data/static_corridor/velodyne")
        dynamic_dest = Path("Backend/data/dynamic_corridor/velodyne")
    else:
        static_dest = Path("data/static_corridor/velodyne")
        dynamic_dest = Path("data/dynamic_corridor/velodyne")

    print(f"Downsampling and copying {len(static_src_bins)} static frames...")
    for idx, f in enumerate(static_src_bins):
        downsample_and_write(f, static_dest / f"{idx:06d}.bin", stride=2)

    print(f"Downsampling and copying {len(dynamic_src_bins)} dynamic frames...")
    for idx, f in enumerate(dynamic_src_bins):
        downsample_and_write(f, dynamic_dest / f"{idx:06d}.bin", stride=2)

    print("Successfully populated static_corridor (50) and dynamic_corridor (70).")


if __name__ == "__main__":
    run_slice()
