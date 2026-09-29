"""
Unit tests for TrenchDetector & Negative Obstacle Analysis (Feature F5.1).
"""

import numpy as np
import pytest

from backend.mapping.cell_statistics import CellStats
from backend.mapping.negative_obstacles import TrenchDetector
from backend.mapping.quadtree import QuadtreeNode


def test_trench_detection():
    detector = TrenchDetector(
        min_elevation_drop=0.25,
        min_radial_gap=0.40,
        num_angular_sectors=36,
        forced_cost=255,
    )

    # 1. Simulate flat ground along forward X-axis (+X direction, angle ~ 0)
    # Points along ray from r = 1.0m to 4.0m at elevation z = 0.0
    r_flat = np.linspace(1.0, 4.0, 30)
    flat_x = r_flat
    flat_y = np.zeros_like(r_flat)
    flat_z = np.zeros_like(r_flat)

    # 2. Sudden trench drop-off: gap from 4.0m to 4.6m (delta_r = 0.6m > 0.40m),
    # followed by trench floor at elevation z = -0.6m (delta_z = 0.6m > 0.25m)
    r_trench = np.linspace(4.6, 7.0, 20)
    trench_x = r_trench
    trench_y = np.zeros_like(r_trench)
    trench_z = np.full_like(r_trench, -0.60)

    # Combine into single point cloud (with dummy side points for polar binning)
    x = np.concatenate([flat_x, trench_x])
    y = np.concatenate([flat_y, trench_y])
    z = np.concatenate([flat_z, trench_z])
    pts = np.column_stack([x, y, z])

    dropoffs = detector.find_dropoffs(pts, sensor_origin=np.array([0.0, 0.0, 1.5]))

    assert len(dropoffs) > 0
    d0 = dropoffs[0]
    assert d0["type"] == "negative_obstacle"
    assert d0["cost"] == 255
    assert d0["drop_depth"] >= 0.25
    # Edge coordinate should be near x = 4.0m
    assert 3.8 <= d0["x"] <= 4.2


def test_apply_dropoffs_to_leaves():
    detector = TrenchDetector(hazard_radius=0.5)

    leaf = QuadtreeNode(x=4.0, y=0.0, size=0.5, depth=0, stats=CellStats(0, 0, 0, 0, 0, 10, 0), cost=0)
    dropoffs = [{"x": 4.1, "y": 0.05, "radius": 0.5, "cost": 255}]

    detector.apply_dropoffs_to_leaves([leaf], dropoffs)
    assert leaf.cost == 255
