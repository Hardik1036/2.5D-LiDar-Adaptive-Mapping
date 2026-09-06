"""
Dataset loader for binary point clouds (.bin, .pcd) and synthetic streams.
Edge-optimized for zero-copy memory layouts where possible.
"""

from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union
import numpy as np

from backend.config import BOUNDS
from backend.utils.synthetic_generator import SyntheticLiDARGenerator

# Optional open3d import for PCD files
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


class DatasetLoader:
    """
    Stream loader providing uniform LiDAR point cloud frames.
    Yields (points, metadata) where points is an (N, 4) float32 array [x, y, z, intensity].
    """

    def __init__(
        self,
        dataset_path: Optional[Union[str, Path]] = None,
        loop: bool = True,
        use_synthetic_fallback: bool = True,
    ):
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.loop = loop
        self.use_synthetic_fallback = use_synthetic_fallback
        self.file_list: List[Path] = []
        self.current_idx = 0
        self.synthetic_gen: Optional[SyntheticLiDARGenerator] = None

        if self.dataset_path and self.dataset_path.exists():
            self._discover_files()
        
        if not self.file_list:
            if self.use_synthetic_fallback:
                self.synthetic_gen = SyntheticLiDARGenerator()
            else:
                raise FileNotFoundError(f"No valid .bin or .pcd files discovered in: {self.dataset_path}")

    def _discover_files(self):
        """Discovers all .bin and .pcd files under the dataset path."""
        if self.dataset_path.is_file():
            if self.dataset_path.suffix.lower() in [".bin", ".pcd"]:
                self.file_list = [self.dataset_path]
        elif self.dataset_path.is_dir():
            # Scan for .bin first (e.g. KITTI / NuScenes)
            bin_files = sorted(list(self.dataset_path.glob("*.bin")))
            if bin_files:
                self.file_list = bin_files
            else:
                pcd_files = sorted(list(self.dataset_path.glob("*.pcd")))
                if pcd_files:
                    self.file_list = pcd_files

    @staticmethod
    def load_bin_file(file_path: Union[str, Path]) -> np.ndarray:
        """
        Parses KITTI/NuScenes binary point cloud (.bin).
        Each point has 4 float32 values: [x, y, z, reflectance].
        """
        raw_data = np.fromfile(str(file_path), dtype=np.float32)
        if raw_data.size % 4 != 0:
            # If 3-channel [x, y, z] without intensity
            if raw_data.size % 3 == 0:
                pts = raw_data.reshape(-1, 3)
                intensity = np.ones((pts.shape[0], 1), dtype=np.float32)
                return np.hstack([pts, intensity])
            raise ValueError(f"Corrupt binary file size {raw_data.size} at {file_path}")
        return raw_data.reshape(-1, 4)

    @staticmethod
    def load_pcd_file(file_path: Union[str, Path]) -> np.ndarray:
        """Parses PCD files using Open3D or lightweight fallback."""
        if HAS_OPEN3D:
            pcd = o3d.io.read_point_cloud(str(file_path))
            pts = np.asarray(pcd.points, dtype=np.float32)
            if pcd.has_colors():
                intensity = np.mean(np.asarray(pcd.colors, dtype=np.float32), axis=1, keepdims=True)
            else:
                intensity = np.ones((pts.shape[0], 1), dtype=np.float32)
            return np.hstack([pts, intensity])
        
        # Simple ASCII PCD fallback parser
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        data_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("DATA"):
                data_idx = i + 1
                break
        coords = []
        for line in lines[data_idx:]:
            tokens = line.strip().split()
            if len(tokens) >= 3:
                coords.append([float(t) for t in tokens[:4]])
        arr = np.array(coords, dtype=np.float32)
        if arr.shape[1] == 3:
            arr = np.hstack([arr, np.ones((arr.shape[0], 1), dtype=np.float32)])
        return arr

    def load_frame(self, frame_idx: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """
        Loads point cloud for a specific frame index or sequential next.
        Filters points strictly within BOUNDS.
        """
        if self.synthetic_gen is not None:
            ts = (self.current_idx * 0.04)
            pts, gt = self.synthetic_gen.generate_frame(timestamp=ts, dt=0.04)
            meta = {
                "source": "synthetic",
                "frame_id": self.current_idx,
                "timestamp": ts,
                "ground_truth": gt,
            }
            self.current_idx += 1
            return pts, meta

        if frame_idx is not None:
            self.current_idx = frame_idx % len(self.file_list)
        elif self.current_idx >= len(self.file_list):
            if self.loop:
                self.current_idx = 0
            else:
                raise StopIteration("End of dataset files reached.")

        target_file = self.file_list[self.current_idx]
        self.current_idx += 1

        if target_file.suffix.lower() == ".bin":
            points = self.load_bin_file(target_file)
        elif target_file.suffix.lower() == ".pcd":
            points = self.load_pcd_file(target_file)
        else:
            raise ValueError(f"Unsupported format: {target_file.suffix}")

        # Crop to spatial bounds
        mask = (
            (points[:, 0] >= BOUNDS.X_MIN) & (points[:, 0] <= BOUNDS.X_MAX) &
            (points[:, 1] >= BOUNDS.Y_MIN) & (points[:, 1] <= BOUNDS.Y_MAX) &
            (points[:, 2] >= BOUNDS.Z_MIN) & (points[:, 2] <= BOUNDS.Z_MAX)
        )
        points = points[mask]

        meta = {
            "source": str(target_file.name),
            "frame_id": self.current_idx - 1,
            "timestamp": (self.current_idx - 1) * 0.04,
            "file_path": str(target_file),
        }
        return points, meta

    def stream(self) -> Generator[Tuple[np.ndarray, dict], None, None]:
        """Generator yielding (points, metadata) continuously."""
        while True:
            try:
                yield self.load_frame()
            except StopIteration:
                break
