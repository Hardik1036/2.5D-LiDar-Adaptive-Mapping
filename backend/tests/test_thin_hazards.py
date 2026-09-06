"""
Unit tests for ThinHazardDetector (Feature F1.6).
"""

import numpy as np
import pytest

from backend.ingestion.thin_hazard_detector import ThinHazardDetector


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
    # Height = 0.025m (2.5 cm), intensity = 0.9 (shiny metallic)
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
