"""
Kaggle Dataset Streaming & Remote Ingestion Adapter for DRISHTI-2.5D.
Streams point cloud sweeps directly from Kaggle into in-memory float32 buffers.
"""

import os
import logging
from pathlib import Path
import time
from typing import Any, Dict, Generator, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("KaggleStreamer")
DEFAULT_DATASET = "hardiknopany/drishti-2-5d-dataset"

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    try:
        import subprocess
        subprocess.check_call(["pip", "install", "kaggle", "kagglehub"])
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        KaggleApi = None


class KaggleStreamer:
    """
    Kaggle dataset synchronization and cache manager for DRISHTI-2.5D.
    Authenticates and downloads point cloud datasets from Kaggle to local disk.
    """

    def __init__(self, dataset_name: Optional[str] = None, cache_dir: str = "data/kaggle_cache"):
        self.dataset_name = (
            dataset_name or os.environ.get("KAGGLE_DATASET") or DEFAULT_DATASET
        )
        resolved_cache = Path(cache_dir)
        if not resolved_cache.is_absolute() and not resolved_cache.exists():
            pkg_cache = Path(__file__).resolve().parent.parent.parent / cache_dir
            if pkg_cache.exists():
                resolved_cache = pkg_cache
        self.cache_dir = resolved_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def sync_dataset(self) -> str:
        """
        Authenticates via Kaggle API and downloads the point cloud dataset if the local cache directory does not contain .bin or point cloud files.
        """
        existing_bins = list(self.cache_dir.rglob("*.bin"))
        if existing_bins:
            logger.info(f"[KaggleStreamer] Found {len(existing_bins)} point clouds in {self.cache_dir}. Skipping download.")
            return str(self.cache_dir)

        username = os.environ.get("KAGGLE_USERNAME")
        key = os.environ.get("KAGGLE_KEY")
        if not username or not key:
            kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
            if not kaggle_json.exists():
                logger.warning("[KaggleStreamer] KAGGLE_USERNAME or KAGGLE_KEY not found in environment.")
                return str(self.cache_dir)

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            logger.info(f"[KaggleStreamer] Authenticated. Downloading {self.dataset_name} to {self.cache_dir}...")
            api.dataset_download_files(self.dataset_name, path=str(self.cache_dir), unzip=True)
            logger.info(f"[KaggleStreamer] Successfully extracted {self.dataset_name} to {self.cache_dir}")
        except Exception as e:
            logger.error(f"[KaggleStreamer] Failed to download dataset: {e}")
        return str(self.cache_dir)


class KaggleDatasetStreamer:

    def __init__(
        self,
        dataset_slug: Optional[str] = None,
        cache_dir: str = "data/kaggle_cache",
        loop: bool = True,
        api: Optional[Any] = None,
    ):
        self.dataset_slug = (
            dataset_slug or os.environ.get("KAGGLE_DATASET") or DEFAULT_DATASET
        )
        # Resolve cache_dir relative to CWD or backend directory
        resolved_cache = Path(cache_dir)
        if not resolved_cache.is_absolute() and not resolved_cache.exists():
            pkg_cache = Path(__file__).resolve().parent.parent.parent / cache_dir
            if pkg_cache.exists():
                resolved_cache = pkg_cache
        self.cache_dir = resolved_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.loop = loop
        self.api = api if api is not None else (KaggleApi() if KaggleApi is not None else None)

        # Ensure environment variables from .env take effect if not already present in os.environ
        try:
            from dotenv import load_dotenv
            backend_env = Path(__file__).resolve().parent.parent.parent / ".env"
            if backend_env.exists():
                load_dotenv(dotenv_path=backend_env)
            load_dotenv()
        except ImportError:
            pass

        # Verify credentials exist before attempting authentication
        if not os.environ.get("KAGGLE_USERNAME") and not (Path.home() / ".kaggle" / "kaggle.json").exists():
            raise RuntimeError(
                "[KaggleStreamer] Missing Kaggle credentials. "
                "Define KAGGLE_USERNAME and KAGGLE_KEY in your root .env file or ~/.kaggle/kaggle.json."
            )

        # Authenticate via environment variables or ~/.kaggle/kaggle.json
        try:
            if self.api is not None:
                self.api.authenticate()
        except Exception as e:
            raise RuntimeError(
                f"[KaggleStreamer] Kaggle authentication failed: {e}. "
                "Ensure KAGGLE_USERNAME and KAGGLE_KEY or ~/.kaggle/kaggle.json exist."
            )

        # Resolve sweeps from cache or fetch from Kaggle
        self.sweep_files: List[Path] = self._resolve_dataset_sweeps()
        if not self.sweep_files:
            raise FileNotFoundError(
                f"[KaggleStreamer] No valid .bin sweeps found in {self.cache_dir}"
            )

    def _resolve_dataset_sweeps(self) -> List[Path]:
        """Inspects cache directory; downloads and extracts archive if sweeps are missing."""
        sweeps = sorted(list(self.cache_dir.rglob("*.bin")))
        if sweeps:
            return sweeps

        print(
            f"[KaggleStreamer] Cache empty. Downloading dataset '{self.dataset_slug}'..."
        )
        try:
            if self.api is not None:
                self.api.dataset_download_files(
                    self.dataset_slug, path=str(self.cache_dir), unzip=True, quiet=False
                )
        except Exception as e:
            # Fallback for alias naming differences (drishti-2-5d-dataset <-> drishti-2-5d-data)
            fallback_slug = None
            if self.dataset_slug.endswith("drishti-2-5d-dataset"):
                fallback_slug = self.dataset_slug.replace("drishti-2-5d-dataset", "drishti-2-5d-data")
            elif self.dataset_slug.endswith("drishti-2-5d-data"):
                fallback_slug = self.dataset_slug.replace("drishti-2-5d-data", "drishti-2-5d-dataset")

            if fallback_slug and self.api is not None:
                try:
                    print(f"[KaggleStreamer] Retrying with alias dataset '{fallback_slug}'...")
                    self.api.dataset_download_files(
                        fallback_slug, path=str(self.cache_dir), unzip=True, quiet=False
                    )
                    self.dataset_slug = fallback_slug
                except Exception:
                    raise RuntimeError(
                        f"[KaggleStreamer] Failed to download dataset '{self.dataset_slug}': {e}. "
                        "Ensure dataset exists, is public or your Kaggle account has permission to access it."
                    ) from e
            else:
                raise RuntimeError(
                    f"[KaggleStreamer] Failed to download dataset '{self.dataset_slug}': {e}. "
                    "Ensure dataset exists, is public or your Kaggle account has permission to access it."
                ) from e

        sweeps = sorted(list(self.cache_dir.rglob("*.bin")))
        return sweeps

    def stream(
        self, fps: float = 20.0
    ) -> Generator[Tuple[np.ndarray, Dict[str, Any]], None, None]:
        """Yields continuous (N, 4) [X, Y, Z, Intensity] normalized point cloud arrays."""
        frame_idx = 0
        num_frames = len(self.sweep_files)
        target_interval = 0.0 if fps <= 0 else (1.0 / max(1.0, fps))

        while True:
            if frame_idx >= num_frames:
                if not self.loop:
                    break
                frame_idx = 0  # Continuous loop over the sequence

            sweep_path = self.sweep_files[frame_idx]
            raw = np.fromfile(str(sweep_path), dtype=np.float32)

            # Handle 4-channel [x, y, z, i] and 5-channel formats
            if raw.size % 4 == 0:
                pts = raw.reshape(-1, 4)
            elif raw.size % 5 == 0:
                pts = raw.reshape(-1, 5)[:, :4]
            else:
                # Fallback for 3-channel points
                pts_3d = raw.reshape(-1, 3)
                intensity = np.full((pts_3d.shape[0], 1), 15.0, dtype=np.float32)
                pts = np.hstack([pts_3d, intensity])

            meta = {
                "frame_id": frame_idx,
                "timestamp": time.time(),
                "filename": sweep_path.name,
                "points_count": len(pts),
            }

            yield pts.astype(np.float32), meta
            frame_idx += 1
            if target_interval > 0:
                time.sleep(target_interval)

    def set_directory(self, new_dir: str) -> bool:
        """Hot-swaps the active .bin sweep directory and resets the sweep pointer."""
        target_path = Path(new_dir)
        if not target_path.exists():
            return False

        new_files = sorted(list(target_path.glob("*.bin")))
        if not new_files:
            new_files = sorted(list(target_path.rglob("*.bin")))

        if new_files:
            self.cache_dir = target_path
            self.sweep_files = new_files
            return True
        return False
