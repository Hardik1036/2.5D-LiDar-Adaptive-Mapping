"""
Local LiDAR Point Cloud Dataset Loader and Sweep Manager.
Provides thread-safe access to raw KITTI .bin sweeps with dynamic directory hot-swapping.
"""

from pathlib import Path
import logging
import threading
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger("DatasetLoader")


class DatasetLoader:
    """
    Manages loading, parsing, and streaming of Velodyne .bin LiDAR sweeps from local disk.
    Supports thread-safe directory hot-swapping, bundled static/dynamic corridors, and continuous looping.
    """

    def __init__(
        self,
        mode_or_dir: Optional[Union[str, Path]] = None,
        mode: Optional[str] = None,
        loop: bool = True,
    ):
        self._lock = threading.Lock()
        self.loop = loop
        self.dataset_dir: Optional[Path] = None
        self.data_dir: Optional[Path] = None
        self.files: List[Path] = []
        self.bin_files: List[Path] = []
        self.current_index: int = 0

        base_data = Path(__file__).resolve().parent.parent.parent / "data"

        target: Optional[Path] = None
        active_mode = (mode or "").lower() if mode else None

        if active_mode is None and isinstance(mode_or_dir, str) and mode_or_dir.lower() in ("static", "dynamic"):
            active_mode = mode_or_dir.lower()
        elif mode_or_dir is not None and not isinstance(mode_or_dir, str):
            target = Path(mode_or_dir)
        elif isinstance(mode_or_dir, str) and mode_or_dir.lower() not in ("static", "dynamic") and (Path(mode_or_dir).exists() or "/" in mode_or_dir or "\\" in mode_or_dir):
            target = Path(mode_or_dir)

        if target is None and (mode_or_dir is not None or mode is not None):
            active_mode = active_mode or "dynamic"
            if active_mode == "static":
                target = base_data / "static_corridor" / "velodyne"
            else:
                target = base_data / "dynamic_corridor" / "velodyne"

            if not target.exists() or len(list(target.glob("*.bin"))) == 0:
                fallback_candidates = [
                    base_data / "kitti_clean" / "training" / "velodyne",
                    base_data / "kaggle_cache" / "data" / "kitti_clean" / "training" / "velodyne",
                    base_data / "test_sweep",
                ]
                for cand in fallback_candidates:
                    if cand.exists() and len(list(cand.glob("*.bin"))) > 0:
                        target = cand
                        break

        if target is not None:
            self.set_directory(str(target))
            self.data_dir = self.dataset_dir
            self.bin_files = self.files
            logger.info(f"[DatasetLoader] Mode: '{active_mode or 'custom'}' | Loaded {len(self.bin_files)} frames from {self.data_dir}")

    def set_directory(self, new_dir: str) -> bool:
        """
        Hot-swaps the active .bin sweep directory and resets the sweep pointer.
        Supports both direct velodyne paths and nested search.
        """
        target_path = Path(new_dir)
        if not target_path.exists():
            logger.warning(f"[DatasetLoader] Target directory does not exist: {target_path}")
            return False

        # Gather .bin files directly or recursively
        new_files = sorted(list(target_path.glob("*.bin")))
        if not new_files:
            new_files = sorted(list(target_path.rglob("*.bin")))

        if new_files:
            with self._lock:
                self.dataset_dir = target_path
                self.data_dir = target_path
                self.files = new_files
                self.bin_files = new_files
                self.current_index = 0
            logger.info(f"[DatasetLoader] Swapped active dataset to {len(self.files)} sweeps in {new_dir}")
            return True
        else:
            logger.warning(f"[DatasetLoader] No .bin sweeps discovered in: {new_dir}")
            return False

    def get_next_sweep(self) -> Optional[np.ndarray]:
        """
        Reads and returns the next point cloud sweep as (N, 4) or (N, 3) float32 array.
        Returns None if no files are available or if sequence has completed (when loop=False).
        """
        with self._lock:
            if not self.files:
                return None
            if self.current_index >= len(self.files):
                if not self.loop:
                    return None
                self.current_index = 0
            file_path = self.files[self.current_index]
            self.current_index += 1

        try:
            raw = np.fromfile(str(file_path), dtype=np.float32)
            if raw.size % 4 == 0:
                pts = raw.reshape(-1, 4)
            elif raw.size % 5 == 0:
                pts = raw.reshape(-1, 5)[:, :4]
            else:
                pts_3d = raw.reshape(-1, 3)
                intensity = np.full((pts_3d.shape[0], 1), 15.0, dtype=np.float32)
                pts = np.hstack([pts_3d, intensity])
            return pts.astype(np.float32)
        except Exception as e:
            logger.warning(f"[DatasetLoader] Failed to read sweep {file_path}: {e}")
            return None

    def get_sweep(self, index: int) -> Optional[np.ndarray]:
        """Retrieves point cloud sweep at a specific index."""
        with self._lock:
            if not self.files or index < 0 or index >= len(self.files):
                return None
            file_path = self.files[index]

        try:
            raw = np.fromfile(str(file_path), dtype=np.float32)
            if raw.size % 4 == 0:
                pts = raw.reshape(-1, 4)
            elif raw.size % 5 == 0:
                pts = raw.reshape(-1, 5)[:, :4]
            else:
                pts_3d = raw.reshape(-1, 3)
                intensity = np.full((pts_3d.shape[0], 1), 15.0, dtype=np.float32)
                pts = np.hstack([pts_3d, intensity])
            return pts.astype(np.float32)
        except Exception as e:
            logger.warning(f"[DatasetLoader] Failed to read sweep at index {index} ({file_path}): {e}")
            return None

    def __len__(self) -> int:
        with self._lock:
            return len(self.files)

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        while True:
            pts = self.get_next_sweep()
            if pts is None:
                break
            yield pts
