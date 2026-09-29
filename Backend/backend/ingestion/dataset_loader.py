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
    Supports thread-safe directory hot-swapping and continuous looping.
    """

    def __init__(
        self,
        dataset_dir: Optional[Union[str, Path]] = None,
        loop: bool = True,
    ):
        self._lock = threading.Lock()
        self.loop = loop
        self.dataset_dir: Optional[Path] = None
        self.files: List[Path] = []
        self.current_index: int = 0

        if dataset_dir:
            self.set_directory(str(dataset_dir))

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
                self.files = new_files
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
