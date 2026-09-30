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
    # Loader automatically resolves candidate paths and loads bundled sweeps
    assert len(loader.files) > 0

    # Test non-existent path properly fails
    assert not loader.set_directory("data/non_existent_dir_12345")

    # Test loading bundled static corridor
    static_path = Path("Backend/data/static_corridor/velodyne")
    if not static_path.exists():
        static_path = Path(__file__).resolve().parent.parent.parent / "data" / "static_corridor" / "velodyne"

    if static_path.exists():
        success = loader.set_directory(str(static_path))
        assert success
        assert len(loader.files) == 50
        assert loader.current_index == 0

        # Read sweep
        pts = loader.get_next_sweep()
        assert pts is not None
        assert isinstance(pts, np.ndarray)
        assert pts.ndim == 2
        assert pts.shape[1] >= 3
        assert loader.current_index == 1

        # Hot-swap to dynamic dataset
        dyn_path = static_path.parent.parent / "dynamic_corridor" / "velodyne"
        if dyn_path.exists():
            success_dyn = loader.set_directory(str(dyn_path))
            assert success_dyn
            assert len(loader.files) == 70
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


def test_dataset_loader_switch_dataset_api():
    loader = DatasetLoader()
    # Switch to dynamic
    res_dyn = loader.switch_dataset("dynamic")
    assert res_dyn is True
    assert "dynamic" in loader.mode
    assert loader.current_idx == 0
    assert loader.current_index == 0
    assert len(loader.files) > 0

    frame1 = loader.get_next_frame()
    assert frame1 is not None
    assert loader.current_index == 1

    # Switch to static
    res_stat = loader.switch_dataset("static")
    assert res_stat is True
    assert "static" in loader.mode
    assert loader.current_index == 0
    assert len(loader.files) > 0

    # Test seek_frame
    target = min(5, len(loader.files) - 1)
    new_idx = loader.seek_frame(target)
    assert new_idx == target
    assert loader.current_index == target
    assert loader.current_idx == target
