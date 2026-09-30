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

# Anchor directory to the absolute path of this file:
CURRENT_FILE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_FILE_DIR.parent.parent
REPO_ROOT = BACKEND_DIR.parent

# Check potential locations where data/sweeps might reside in Docker / Railway / Local environments
CANDIDATE_PATHS = [
    # Bundled real sweeps (tracked in git)
    BACKEND_DIR / "data" / "dynamic_corridor" / "velodyne",
    BACKEND_DIR / "data" / "static_corridor" / "velodyne",
    REPO_ROOT / "Backend" / "data" / "dynamic_corridor" / "velodyne",
    REPO_ROOT / "Backend" / "data" / "static_corridor" / "velodyne",
    REPO_ROOT / "data" / "dynamic_corridor" / "velodyne",
    REPO_ROOT / "data" / "static_corridor" / "velodyne",
    Path("/app/data/dynamic_corridor/velodyne"),
    Path("/app/Backend/data/dynamic_corridor/velodyne"),
    Path("/app/backend/data/dynamic_corridor/velodyne"),
    Path("/app/data/static_corridor/velodyne"),
    Path("/app/Backend/data/static_corridor/velodyne"),
    # General sweeps folders as specified in directive
    CURRENT_FILE_DIR.parent / "data" / "sweeps",
    BACKEND_DIR / "data" / "sweeps",
    REPO_ROOT / "data" / "sweeps",
    Path("/app/data/sweeps"),
    Path("/app/backend/data/sweeps"),
    Path("/app/Backend/data/sweeps"),
]


def resolve_sweep_directory(
    preferred_mode: Optional[str] = None,
    candidate_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Resolves the active LiDAR sweeps directory across Docker, Railway,
    and local paths, ensuring real .bin sweeps are discovered.
    """
    # 1. If explicit candidate directory is passed and contains .bin files, use it
    if candidate_dir:
        cand_path = Path(candidate_dir)
        if cand_path.exists():
            if any(cand_path.glob("*.bin")):
                logger.info(f"[DatasetLoader] Found valid sweeps directory at: {cand_path.resolve()}")
                return cand_path
            # Check nested velodyne
            if (cand_path / "velodyne").exists() and any((cand_path / "velodyne").glob("*.bin")):
                logger.info(f"[DatasetLoader] Found valid sweeps directory at: {(cand_path / 'velodyne').resolve()}")
                return cand_path / "velodyne"
            if (cand_path / "training" / "velodyne").exists() and any((cand_path / "training" / "velodyne").glob("*.bin")):
                logger.info(f"[DatasetLoader] Found valid sweeps directory at: {(cand_path / 'training' / 'velodyne').resolve()}")
                return cand_path / "training" / "velodyne"

    mode = (preferred_mode or "dynamic").lower()

    # Prioritize mode-specific paths
    if mode == "static":
        mode_candidates = [p for p in CANDIDATE_PATHS if "static" in str(p)]
        other_candidates = [p for p in CANDIDATE_PATHS if "static" not in str(p)]
    else:
        mode_candidates = [p for p in CANDIDATE_PATHS if "dynamic" in str(p)]
        other_candidates = [p for p in CANDIDATE_PATHS if "dynamic" not in str(p)]

    for candidate in mode_candidates + other_candidates:
        if candidate.exists() and any(candidate.glob("*.bin")):
            logger.info(f"[DatasetLoader] Found valid sweeps directory at: {candidate.resolve()}")
            return candidate

    # Recursive fallback across common data search roots
    for root in [
        BACKEND_DIR / "data",
        REPO_ROOT / "data",
        REPO_ROOT / "Backend" / "data",
        Path("/app/data"),
        Path("/app/Backend/data"),
        Path("data"),
    ]:
        if root.exists():
            bins = list(root.rglob("*.bin"))
            if bins:
                parent_dir = bins[0].parent
                logger.info(f"[DatasetLoader] Found valid sweeps directory via rglob at: {parent_dir.resolve()}")
                return parent_dir

    # Fallback to the primary candidate if none exist yet
    fallback = BACKEND_DIR / "data" / "dynamic_corridor" / "velodyne"
    logger.warning(f"[DatasetLoader] Could not find .bin files in candidates. Defaulting to: {fallback}")
    return fallback


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
        self.current_idx: int = 0

        active_mode = "static"
        candidate_dir = None

        if isinstance(mode_or_dir, str) and mode_or_dir.lower() in ("static", "dynamic", "static_corridor", "dynamic_corridor"):
            active_mode = mode_or_dir.lower()
        elif mode_or_dir is not None:
            candidate_dir = mode_or_dir

        if mode is not None:
            active_mode = mode.lower()

        self.mode = active_mode

        # 1. Anchor to this file: Backend/backend/ingestion/dataset_loader.py
        # .parent = ingestion, .parent.parent = backend, .parent.parent.parent = Backend (or /app in Docker)
        BASE_DIR = Path(__file__).resolve().parent.parent.parent

        # Resolve folder name (support both 'static' and 'static_corridor')
        mode_folder = f"{self.mode}_corridor" if not str(self.mode).endswith("_corridor") else self.mode

        # Primary container-safe data path
        data_path = BASE_DIR / "data" / mode_folder / "velodyne"
        if not data_path.exists() or not any(data_path.glob("*.bin")):
            # Check direct mode name without _corridor
            alt_path = BASE_DIR / "data" / self.mode / "velodyne"
            if alt_path.exists() and any(alt_path.glob("*.bin")):
                data_path = alt_path
            else:
                # Use resolve_sweep_directory across all fallback and Docker candidate paths
                data_path = resolve_sweep_directory(preferred_mode=self.mode, candidate_dir=candidate_dir)

        self.data_dir = data_path
        self.dataset_dir = data_path
        self.sweep_dir = data_path
        self.files = sorted(list(self.data_dir.glob("*.bin"))) if self.data_dir.exists() else []
        self.bin_files = self.files

        # 2. Add loud debugging so we can see the exact path in Railway logs
        print(f"\n[DatasetLoader] BOOTSTRAP PATH RESOLUTION:")
        print(f" -> Looking for sweeps in: {self.data_dir}")
        print(f" -> Found {len(self.files)} .bin files.")

        if len(self.files) == 0:
            print(f"[CRITICAL WARNING] FALLING BACK TO SYNTHETIC DATA!")
            print(f" -> Verify that Railway's 'Root Directory' contains the 'data/' folder.")
        else:
            logger.info(f"[DatasetLoader] Mode: '{self.mode}' | Loaded {len(self.bin_files)} frames from {self.data_dir}")

    def set_directory(self, new_dir: str) -> bool:
        """
        Hot-swaps the active .bin sweep directory and resets the sweep pointer.
        Supports both direct velodyne paths and nested search.
        """
        target_path = Path(new_dir)
        if not target_path.exists():
            # Check relative to known project roots
            for base in [BACKEND_DIR, REPO_ROOT, Path("/app"), Path("/app/Backend")]:
                cand = base / new_dir
                if cand.exists():
                    target_path = cand
                    break

        if not target_path.exists():
            logger.warning(f"[DatasetLoader] Target directory does not exist: {target_path}")
            return False

        if (target_path / "velodyne").exists() and any((target_path / "velodyne").glob("*.bin")):
            target_path = target_path / "velodyne"
        elif (target_path / "training" / "velodyne").exists() and any((target_path / "training" / "velodyne").glob("*.bin")):
            target_path = target_path / "training" / "velodyne"

        # Gather .bin files directly or recursively
        new_files = sorted(list(target_path.glob("*.bin")))
        if not new_files:
            new_files = sorted(list(target_path.rglob("*.bin")))

        if new_files:
            with self._lock:
                self.dataset_dir = target_path
                self.data_dir = target_path
                self.sweep_dir = target_path
                self.files = new_files
                self.bin_files = new_files
                self.current_index = 0
                self.current_idx = 0
            logger.info(f"[DatasetLoader] Swapped active dataset to {len(self.files)} sweeps in {target_path}")
            return True
        else:
            logger.warning(f"[DatasetLoader] No .bin sweeps discovered in: {new_dir}")
            return False

    def resolve_sweep_directory(self, preferred_mode: Optional[str] = None) -> Path:
        """Resolves the active LiDAR sweeps directory for the current or preferred mode."""
        mode = preferred_mode or self.mode
        return resolve_sweep_directory(preferred_mode=mode, candidate_dir=self.data_dir)

    def switch_dataset(self, mode: str) -> bool:
        """
        Safely swaps the dataset mode between 'static' and 'dynamic' corridor sweeps
        without setting file lists to empty or raising IndexError.
        """
        with self._lock:
            self.mode = "dynamic_corridor" if "dynamic" in str(mode).lower() else "static_corridor"
            resolved_dir = self.resolve_sweep_directory()
            self.data_dir = resolved_dir
            self.dataset_dir = resolved_dir
            self.sweep_dir = resolved_dir
            if resolved_dir.exists():
                new_files = sorted(list(resolved_dir.glob("*.bin")))
                if not new_files:
                    new_files = sorted(list(resolved_dir.rglob("*.bin")))
            else:
                new_files = []

            self.files = new_files
            self.bin_files = new_files
            self.current_index = 0
            self.current_idx = 0
            logger.info(f"[DatasetLoader] Swapped to {self.mode} with {len(self.files)} sweeps from {self.data_dir}.")
            return True

    def get_next_frame(self) -> Optional[np.ndarray]:
        """Alias for get_next_sweep for compatibility with frame stepping commands."""
        return self.get_next_sweep()

    def seek_frame(self, target_frame: int) -> int:
        """Safely updates current_index / current_idx to the specified frame."""
        with self._lock:
            if not self.files:
                self.current_index = 0
                self.current_idx = 0
                return 0
            idx = max(0, min(int(target_frame), len(self.files) - 1))
            self.current_index = idx
            self.current_idx = idx
            return idx

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
            self.current_idx = self.current_index

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
