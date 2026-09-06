"""
Unit tests for ThinHazardDetector (Feature F1.6) with ThreatNet1D ONNX inference.
"""

from pathlib import Path
import numpy as np
import pytest

from backend.ingestion.thin_hazard_detector import ThinHazardDetector
from backend.mapping.quadtree import AdaptiveQuadtree, QuadtreeNode


def test_thin_hazard_detection():
    detector = ThinHazardDetector(
        min_residual=0.015,
        max_residual=0.050,
        min_points_cluster=5,
        forced_cost=255,
    )

    # 1. Base ground plane at z = 0.0 with typical terrain intensity ~ 0.3
    gx, gy = np.meshgrid(np.linspace(-3, 3, 30), np.linspace(-3, 3, 30))
    n_ground = gx.size
    ground_pts = np.column_stack([
        gx.flatten(),
        gy.flatten(),
        np.zeros(n_ground, dtype=np.float32),
        np.full(n_ground, 0.3, dtype=np.float32),
    ])

    # 2. Inject a 2.5 cm high metallic spike strip / plate at x in [0.5, 0.7], y in [-0.5, 0.5]
    # Height = 0.025m (2.5 cm), intensity = 0.95 (shiny metallic)
    sx, sy = np.meshgrid(np.linspace(0.5, 0.7, 5), np.linspace(-0.5, 0.5, 10))
    n_strip = sx.size
    strip_pts = np.column_stack([
        sx.flatten(),
        sy.flatten(),
        np.full(n_strip, 0.025, dtype=np.float32),
        np.full(n_strip, 0.95, dtype=np.float32),
    ])

    all_ground = np.vstack([ground_pts, strip_pts])

    hazards = detector.detect_low_profile_hazards(all_ground)

    assert len(hazards) > 0
    h0 = hazards[0]
    assert h0["cost"] == 255
    assert h0["type"] == "thin_hazard"
    # Centroid should be around x=0.6, y=0.0
    assert 0.4 <= h0["centroid"][0] <= 0.8
    assert -0.6 <= h0["centroid"][1] <= 0.6


def test_thin_hazard_empty():
    detector = ThinHazardDetector()
    assert detector.detect_low_profile_hazards(np.empty((0, 4))) == []


def test_threatnet1d_model_and_calibration_loading():
    """Verifies ThreatNet1D ONNX session loads properly and parses calibration threshold 0.75."""
    detector = ThinHazardDetector()
    assert detector.session is not None, "ThreatNet1D ONNX session should initialize successfully"
    assert pytest.approx(detector.threshold, abs=1e-3) == 0.75, "Calibration threshold should be 0.75"


def test_threatnet1d_spike_strip_probability_above_threshold():
    """Verifies synthetic spike strip inputs evaluate to > 0.75 probability on ThreatNet1D."""
    detector = ThinHazardDetector(min_residual=0.010, max_residual=0.080)

    # 1. Base terrain ground points: z = 0.0, intensity = 0.2
    gx, gy = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
    n_g = gx.size
    ground = np.column_stack([
        gx.flatten(),
        gy.flatten(),
        np.zeros(n_g, dtype=np.float32),
        np.full(n_g, 0.2, dtype=np.float32),
    ])

    # 2. Synthetic spike strip: delta_z = 0.035m (3.5 cm), metallic reflectance = 0.95
    sx, sy = np.meshgrid(np.linspace(1.0, 1.2, 3), np.linspace(0.0, 0.5, 4))
    n_s = sx.size
    spikes = np.column_stack([
        sx.flatten(),
        sy.flatten(),
        np.full(n_s, 0.035, dtype=np.float32),
        np.full(n_s, 0.95, dtype=np.float32),
    ])

    pts = np.vstack([ground, spikes])

    cand_indices, threat_probs = detector.predict_probabilities(pts)
    assert len(cand_indices) > 0, "Candidate near-ground points must be identified"
    max_prob = float(np.max(threat_probs))
    assert max_prob > 0.75, f"Spike strip probability {max_prob:.4f} must exceed calibrated threshold (0.75)"

    # Confirm detect_threats flags it with cost = 255
    hazards = detector.detect_threats(pts)
    assert len(hazards) > 0
    assert hazards[0]["cost"] == 255
    assert hazards[0]["threat_probability"] > 0.75


def test_apply_hazards_to_leaves():
    """Verifies that apply_hazards_to_leaves directly marks quadtree leaf node costs to 255."""
    detector = ThinHazardDetector(forced_cost=255)

    leaf1 = QuadtreeNode(x=1.0, y=1.0, size=0.5, depth=1, cost=0)
    leaf2 = QuadtreeNode(x=5.0, y=5.0, size=0.5, depth=1, cost=0)
    leaves = [leaf1, leaf2]

    hazards = [{
        "type": "thin_hazard",
        "centroid": (1.05, 1.05, 0.03),
        "radius": 0.3,
        "cost": 255,
    }]

    detector.apply_hazards_to_leaves(leaves, hazards)
    assert leaf1.cost == 255, "Leaf 1 overlapping hazard must have cost = 255"
    assert leaf2.cost == 0, "Leaf 2 distant from hazard must maintain cost = 0"


def test_numpy_fallback():
    """Verifies that ThinHazardDetector functions with fast NumPy fallback when ONNX is disabled."""
    detector = ThinHazardDetector(use_onnx=False)
    assert detector.session is None

    # Inject high contrast spike points
    ground = np.array([
        [0.0, 0.0, 0.0, 0.2],
        [1.0, 0.0, 0.0, 0.2],
        [0.0, 1.0, 0.0, 0.2],
        [1.0, 1.0, 0.0, 0.2],
        # Spike cluster
        [0.5, 0.5, 0.035, 0.95],
        [0.52, 0.5, 0.035, 0.95],
        [0.54, 0.5, 0.035, 0.95],
        [0.56, 0.5, 0.035, 0.95],
        [0.58, 0.5, 0.035, 0.95],
    ], dtype=np.float32)

    cand_indices, probs = detector.predict_probabilities(ground)
    assert len(cand_indices) > 0
    assert len(probs) == len(cand_indices)
