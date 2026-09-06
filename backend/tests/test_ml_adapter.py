"""
Unit tests for MLPerceptionAdapter.
"""

import numpy as np
import pytest

from backend.adapters.ml_adapter import MLPerceptionAdapter
from backend.tracking.clustering import DetectedCluster


def test_process_semantic_labels():
    adapter = MLPerceptionAdapter()

    # Create 100 points
    points = np.random.uniform(-10.0, 10.0, size=(100, 3)).astype(np.float32)
    labels = np.zeros(100, dtype=np.int32)
    labels[0:40] = 0   # Ground
    labels[40:60] = 1  # Vegetation
    labels[60:85] = 2  # Rigid
    labels[85:100] = 3 # Dynamic

    res = adapter.process_semantic_labels(points, labels)

    assert len(res["ground"]) == 40
    assert len(res["vegetation"]) == 20
    assert len(res["rigid"]) == 25
    assert len(res["dynamic"]) == 15
    assert len(res["obstacle"]) == 40  # 25 + 15

    # Check empty input handling
    empty_res = adapter.process_semantic_labels(np.empty((0, 3)), np.array([]))
    assert len(empty_res["ground"]) == 0
    assert len(empty_res["obstacle"]) == 0


def test_process_pillar_detections():
    adapter = MLPerceptionAdapter()

    # [x, y, z, l, w, h, yaw, vx, vy]
    dets = np.array([
        [5.0, 3.0, 0.5, 4.0, 2.0, 1.5, 0.2, 1.2, -0.5],
        [-2.0, -4.0, 0.0, 2.5, 1.0, 1.8, -0.1, 0.0, 0.0],
    ], dtype=np.float32)

    clusters = adapter.process_pillar_detections(dets)

    assert len(clusters) == 2
    c0 = clusters[0]
    assert isinstance(c0, DetectedCluster)
    assert c0.centroid == (5.0, 3.0, 0.5)
    assert c0.dimensions == (4.0, 2.0, 1.5)
    assert c0.bbox == (3.0, 7.0, 2.0, 4.0, -0.25, 1.25)
    assert hasattr(c0, "velocity")
    assert getattr(c0, "velocity") == (pytest.approx(1.2), pytest.approx(-0.5))

    # Test empty detections
    assert adapter.process_pillar_detections(np.empty((0, 9))) == []
