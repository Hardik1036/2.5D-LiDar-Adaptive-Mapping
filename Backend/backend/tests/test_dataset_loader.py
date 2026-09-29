"""
Unit tests for DatasetLoader and WebSocket set_dataset runtime switching.
"""

from pathlib import Path
import pytest
import numpy as np

from backend.ingestion.dataset_loader import DatasetLoader
from backend.main import PerceptionPipeline


def test_dataset_loader_initialization_and_hot_swap():
    loader = DatasetLoader()
    assert len(loader.files) == 0

    # Test non-existent path
    assert not loader.set_directory("data/non_existent_dir_12345")

    # Test loading clean dataset
    clean_path = Path("data/kaggle_cache/data/kitti_clean/training/velodyne")
    if not clean_path.exists():
        clean_path = Path(__file__).resolve().parent.parent.parent / "data" / "kaggle_cache" / "data" / "kitti_clean" / "training" / "velodyne"

    if clean_path.exists():
        success = loader.set_directory(str(clean_path))
        assert success
        assert len(loader.files) > 0
        assert loader.current_index == 0

        # Read sweep
        pts = loader.get_next_sweep()
        assert pts is not None
        assert isinstance(pts, np.ndarray)
        assert pts.ndim == 2
        assert pts.shape[1] >= 3
        assert loader.current_index == 1

        # Hot-swap to dynamic dataset
        dyn_path = clean_path.parent.parent.parent / "kitti_dynamic" / "training" / "velodyne"
        if dyn_path.exists():
            success_dyn = loader.set_directory(str(dyn_path))
            assert success_dyn
            assert len(loader.files) > 0
            assert loader.current_index == 0
            pts_dyn = loader.get_next_sweep()
            assert pts_dyn is not None


def test_pipeline_set_dataset_command_handler():
    pipeline = PerceptionPipeline()
    assert hasattr(pipeline, "dataset_loader")
    assert hasattr(pipeline, "current_dataset_mode")

    # Test switching to dynamic
    pipeline.handle_client_command({"action": "set_dataset", "mode": "dynamic"})
    assert pipeline.current_dataset_mode == "dynamic"
    assert len(pipeline.dataset_loader.files) > 0

    # Test switching back to static
    pipeline.handle_client_command({"action": "set_dataset", "mode": "static"})
    assert pipeline.current_dataset_mode == "static"
    assert len(pipeline.dataset_loader.files) > 0
