"""
Unit tests for MLPerceptionAdapter and ThreatNet1D thin hazard inference.
"""

import numpy as np
import pytest

from backend.adapters.ml_adapter import MLPerceptionAdapter, evaluate_thin_hazards
from backend.ingestion.ml_adapter import evaluate_thin_hazards as evaluate_thin_hazards_ingestion
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


def test_evaluate_thin_hazards():
    """Verifies that evaluate_thin_hazards executes ThreatNet1D and flags synthetic spike strip inputs > 0.75."""
    adapter = MLPerceptionAdapter()

    # 1. Nominal terrain ground points
    gx, gy = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
    n_g = gx.size
    ground = np.column_stack([
        gx.flatten(),
        gy.flatten(),
        np.zeros(n_g, dtype=np.float32),
        np.full(n_g, 0.2, dtype=np.float32),
    ])

    # 2. Inject synthetic spike strip: delta_z = 0.035m (3.5 cm), metallic reflectance = 0.95
    sx, sy = np.meshgrid(np.linspace(1.0, 1.2, 3), np.linspace(0.0, 0.5, 4))
    n_s = sx.size
    spikes = np.column_stack([
        sx.flatten(),
        sy.flatten(),
        np.full(n_s, 0.035, dtype=np.float32),
        np.full(n_s, 0.95, dtype=np.float32),
    ])

    all_pts = np.vstack([ground, spikes])

    # Test instance method
    res = adapter.evaluate_thin_hazards(all_pts)
    assert "threat_mask" in res
    assert "probabilities" in res
    assert "threat_points" in res
    assert "threshold" in res
    assert pytest.approx(res["threshold"], abs=1e-3) == 0.75

    max_prob = float(np.max(res["probabilities"]))
    assert max_prob > 0.75, f"Spike strip probability {max_prob:.4f} must exceed threshold (0.75)"
    assert np.any(res["threat_mask"])
    assert len(res["threat_points"]) > 0

    # Test ingestion re-export function
    res_ingest = evaluate_thin_hazards_ingestion(all_pts)
    assert np.array_equal(res["threat_mask"], res_ingest["threat_mask"])


def test_evaluate_thin_hazards_empty():
    """Verifies that evaluate_thin_hazards gracefully handles empty arrays."""
    res = evaluate_thin_hazards(np.empty((0, 4)))
    assert len(res["threat_mask"]) == 0
    assert len(res["probabilities"]) == 0
    assert len(res["threat_points"]) == 0
