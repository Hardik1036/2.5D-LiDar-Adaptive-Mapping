"""
Unit tests for SensorHealthMonitor & Degraded Mode Safe Fallback (Feature F5.3).
"""

import numpy as np
import pytest

from backend.mapping.cell_statistics import CellStats
from backend.mapping.degraded_mode import SensorHealthMonitor
from backend.mapping.quadtree import QuadtreeNode
from backend.mapping.temporal_blender import TemporalMapBlender


def test_sensor_health_monitor_nominal_and_degraded():
    monitor = SensorHealthMonitor(min_baseline_pts=50, drop_threshold_ratio=0.15)

    # 1. Feed 10 frames of healthy symmetric point clouds
    gx, gy = np.meshgrid(np.linspace(-5, 5, 20), np.linspace(-5, 5, 20))
    nominal_pts = np.column_stack([gx.flatten(), gy.flatten(), np.zeros(gx.size)])

    for _ in range(10):
        health = monitor.update(nominal_pts)

    assert health["status"] == SensorHealthMonitor.STATUS_HEALTHY
    assert health["system_status"] == SensorHealthMonitor.ALERT_NOMINAL
    assert health["is_degraded"] is False

    # 2. Simulate forward-right (quadrant 0: x > 0, y > 0) lens occlusion / smoke blockage
    # Strip all points with x > 0 and y > 0
    occluded_mask = ~((nominal_pts[:, 0] > 0) & (nominal_pts[:, 1] > 0))
    occluded_pts = nominal_pts[occluded_mask]

    degraded_health = monitor.update(occluded_pts)

    assert degraded_health["status"] == SensorHealthMonitor.STATUS_DEGRADED
    assert degraded_health["system_status"] == SensorHealthMonitor.ALERT_CAUTION_ACTIVE
    assert degraded_health["is_degraded"] is True
    assert "FRONT_RIGHT" in degraded_health["degraded_quadrants"]


def test_degraded_costmap_inflation_and_memory():
    monitor = SensorHealthMonitor(min_baseline_pts=50, drop_threshold_ratio=0.15)
    blender = TemporalMapBlender()

    # Pre-train baseline
    gx, gy = np.meshgrid(np.linspace(-5, 5, 20), np.linspace(-5, 5, 20))
    nominal_pts = np.column_stack([gx.flatten(), gy.flatten(), np.zeros(gx.size)])
    for _ in range(8):
        monitor.update(nominal_pts)

    # Occlude Front-Right (Q0)
    occluded_pts = nominal_pts[~((nominal_pts[:, 0] > 0) & (nominal_pts[:, 1] > 0))]
    monitor.update(occluded_pts)

    # Create test leaf in Front-Right (x = 3.0, y = 3.0) with cost 0
    leaf_fr = QuadtreeNode(x=3.0, y=3.0, size=0.5, depth=0, stats=CellStats(0, 0, 0, 0, 0, 10, mean_z=0.0), cost=0)
    # Create test leaf in Front-Left (x = -3.0, y = 3.0) with cost 0
    leaf_fl = QuadtreeNode(x=-3.0, y=3.0, size=0.5, depth=0, stats=CellStats(0, 0, 0, 0, 0, 10, mean_z=0.0), cost=0)

    leaves = [leaf_fr, leaf_fl]

    # Populate blender memory with historical ground elevation (z = -1.5m)
    blender.update_cell(3.0, 3.0, -1.50)

    # Apply costmap inflation
    monitor.apply_degraded_costmap_inflation(leaves, caution_cost=180)
    assert leaf_fr.cost == 180  # Inflated to caution
    assert leaf_fl.cost == 0    # Left nominal

    # Preserve terrain memory
    monitor.preserve_terrain_memory(leaves, blender)
    assert leaf_fr.stats.mean_z == pytest.approx(-1.50)
