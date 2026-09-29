"""
Unit tests for TemporalMapBlender (Feature F2.4).
"""

import numpy as np
import pytest

from backend.mapping.cell_statistics import CellStats
from backend.mapping.quadtree import QuadtreeNode
from backend.mapping.temporal_blender import TemporalMapBlender


def test_temporal_map_blender():
    blender = TemporalMapBlender(grid_resolution=0.10, sensor_noise_sigma=0.05)

    # 1. Update same cell repeatedly with noisy observations around z = 1.0m
    z_true = 1.0
    measurements = [1.10, 0.92, 1.05, 0.98, 1.02]

    smoothed_vals = []
    for m in measurements:
        s = blender.update_cell(x=2.5, y=3.5, z_meas=m)
        smoothed_vals.append(s)

    # Kalman filter should converge near 1.0m and exhibit lower variance than raw inputs
    assert abs(smoothed_vals[-1] - z_true) < 0.05
    assert np.var(smoothed_vals) < np.var(measurements)


def test_blend_quadtree():
    blender = TemporalMapBlender()
    leaf = QuadtreeNode(x=1.0, y=1.0, size=0.5, depth=0, stats=CellStats(0, 0, 0, 0, 0, 10, mean_z=0.5))

    blender.blend_quadtree([leaf])
    assert leaf.stats.mean_z == pytest.approx(0.5)

    # Second frame with noise
    leaf.stats.mean_z = 0.65
    blender.blend_quadtree([leaf])
    # Should be pulled between 0.5 and 0.65
    assert 0.50 < leaf.stats.mean_z < 0.65
