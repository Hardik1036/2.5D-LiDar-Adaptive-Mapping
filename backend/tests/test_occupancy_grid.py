"""
Unit tests for ROS 2 OccupancyGrid builder, Vegetation Filter, and Database Adapter.
"""

import numpy as np
import pytest

from backend.adapters.db_adapter import DatabaseAdapter
from backend.config import NAV2
from backend.mapping.cell_statistics import CellStats
from backend.mapping.occupancy_publisher import OccupancyGridBuilder
from backend.mapping.quadtree import QuadtreeNode
from backend.mapping.vegetation_filter import VegetationFilter


def test_occupancy_grid_builder():
    builder = OccupancyGridBuilder()

    # Create synthetic leaves with different costs
    node_free = QuadtreeNode(x=0.0, y=0.0, size=1.0, depth=0, stats=CellStats(0, 0, 0, 0, 0, 10, 0), cost=10)
    node_caution = QuadtreeNode(x=5.0, y=5.0, size=1.0, depth=0, stats=CellStats(0, 0, 0, 0, 0, 10, 0), cost=120)
    node_lethal = QuadtreeNode(x=-5.0, y=-5.0, size=1.0, depth=0, stats=CellStats(0, 0, 0, 0, 0, 10, 0), cost=255)

    leaves = [node_free, node_caution, node_lethal]
    hazards = [{"x": 2.0, "y": 2.0, "radius": 1.0, "cost": 255}]

    grid = builder.build_grid(leaves, hazards)

    assert grid.shape == (NAV2.HEIGHT, NAV2.WIDTH)
    assert grid.dtype == np.int8

    # Check that free, caution, lethal values exist in the grid
    assert np.any(grid == NAV2.COST_FREE)
    assert np.any(grid == NAV2.COST_CAUTION)
    assert np.any(grid == NAV2.COST_LETHAL)
    assert np.any(grid == NAV2.COST_UNKNOWN)

    # Check ROS message serialization
    ros_msg = builder.to_ros_message_dict(grid, timestamp=100.5, flatten_data=False)
    assert ros_msg["header"]["frame_id"] == "map"
    assert ros_msg["header"]["stamp"]["sec"] == 100
    assert ros_msg["info"]["width"] == NAV2.WIDTH
    assert ros_msg["info"]["height"] == NAV2.HEIGHT
    assert ros_msg["info"]["resolution"] == pytest.approx(NAV2.RESOLUTION)


def test_vegetation_filter():
    veg_filter = VegetationFilter(max_vegetation_cost=20, lethal_cost=255)

    # Create high-cost leaf (e.g. geometric variance triggered high cost)
    leaf_veg = QuadtreeNode(x=2.0, y=2.0, size=0.5, depth=0, stats=CellStats(0, 0, 0, 0, 0, 10, 0), cost=180)
    leaf_rigid = QuadtreeNode(x=-2.0, y=-2.0, size=0.5, depth=0, stats=CellStats(0, 0, 0, 0, 0, 10, 0), cost=10)

    leaves = [leaf_veg, leaf_rigid]

    veg_pts = np.array([[2.0, 2.0, 0.5]], dtype=np.float32)
    rigid_pts = np.array([[-2.0, -2.0, 0.5]], dtype=np.float32)

    veg_filter.apply_vegetation_scaling(leaves, vegetation_points=veg_pts, rigid_points=rigid_pts)

    assert leaf_veg.cost == 20     # Capped down from 180 to 20
    assert leaf_rigid.cost == 255  # Promoted to lethal 255


def test_database_adapter_fallback():
    # Test local memory fallback when Redis is offline
    db = DatabaseAdapter(enabled=False)
    assert not db.is_connected

    # Test pose set & get
    db.set_vehicle_pose(1.5, 2.5, 0.3, 0.78)
    pose = db.get_vehicle_pose()
    assert pose[0] == pytest.approx(1.5)
    assert pose[1] == pytest.approx(2.5)
    assert pose[2] == pytest.approx(0.3)
    assert pose[3] == pytest.approx(0.78)

    # Test tracks sync
    dummy_tracks = [{"id": 1, "x": 5.0, "y": 6.0, "speed": 1.2}]
    db.sync_active_tracks(dummy_tracks)
    retrieved = db.get_active_tracks()
    assert len(retrieved) == 1
    assert retrieved[0]["id"] == 1

    # Test telemetry logging
    db.log_frame_telemetry({"fps": 35.2, "latency_ms": 22.1})
    assert len(db._memory_store[db.config.KEY_TELEMETRY]) == 1
