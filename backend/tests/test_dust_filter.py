"""
Unit tests for StatisticalDustFilter (Feature F1.3).
"""

import numpy as np
import pytest

from backend.ingestion.dust_filter import StatisticalDustFilter


def test_dust_filter_removes_scatter():
    filter_mod = StatisticalDustFilter(k=10, std_ratio=2.0)

    # 1. Create a dense grid of points (e.g., ground plane)
    gx, gy = np.meshgrid(np.linspace(-2, 2, 25), np.linspace(-2, 2, 25))
    dense_pts = np.column_stack([gx.flatten(), gy.flatten(), np.zeros(gx.size)])

    # 2. Inject sparse isolated dust points in the air far away
    dust_pts = np.array([
        [10.0, 10.0, 5.0],
        [-15.0, 8.0, 4.0],
        [8.0, -12.0, 6.0],
        [0.0, 0.0, 10.0],
    ], dtype=np.float32)

    all_pts = np.vstack([dense_pts, dust_pts])

    filtered = filter_mod.filter(all_pts, k=10, std_ratio=2.0)

    # Check that dust points were stripped while dense structure is preserved
    assert len(filtered) < len(all_pts)
    assert len(filtered) >= len(dense_pts) - 5

    # Check none of the extreme dust points remain
    for d in dust_pts:
        dists = np.linalg.norm(filtered[:, :3] - d, axis=1)
        assert np.min(dists) > 0.1, f"Dust point {d} was not filtered out"


def test_dust_filter_empty_and_small():
    filter_mod = StatisticalDustFilter()

    # Empty array
    assert len(filter_mod.filter(np.empty((0, 3)))) == 0

    # Very small point set (less than k)
    small_pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    res = filter_mod.filter(small_pts)
    assert len(res) == 1
