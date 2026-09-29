"""
Unit tests for Width-Aware Ground Coarsening & Clearance-Based Quadtree Sizing.
DRISHTI-2.5D Adaptive Mapping (SIH26053 / DRDO PS-53)
"""

import numpy as np
import pytest
from backend.config import BOUNDS
from backend.mapping.quadtree import AdaptiveQuadtree, QuadtreeNode
from backend.mapping.cell_statistics import CellStats


def test_wide_drivable_corridor_merges_to_1m_super_tiles():
    """
    Verifies that open drivable space with lateral clearance >= 1.0m
    (total free width > 2.0m) merges 2x2 clusters of 0.50m cells
    into a single 1.0m x 1.0m super-tile (Cost = 0).
    """
    qt = AdaptiveQuadtree()
    x_min = qt.bounds.X_MIN
    y_min = qt.bounds.Y_MIN
    res = qt.coarse_res

    # Create an isolated 4x4 block of ground cells far from any obstacles
    # Grid coordinates gx in [10, 13], gy in [10, 13] (size 2.0m x 2.0m)
    # The inner 2x2 cells at [11, 12] x [11, 12] have clearance >= 1.0m
    # To provide clearance >= 1.0m, create an 8x8 ground patch
    leaves = []
    for gy in range(8, 16):
        for gx in range(8, 16):
            cx = x_min + (gx + 0.5) * res
            cy = y_min + (gy + 0.5) * res
            st = CellStats(
                z_min=-1.62,
                z_max=-1.58,
                delta_z=0.04,
                variance=0.0001,
                slope=0.0,
                point_count=10,
                mean_z=-1.60,
            )
            node = QuadtreeNode(x=cx, y=cy, size=res, depth=0, stats=st, cost=0, is_leaf=True)
            leaves.append(node)

    qt.leaves = leaves
    coarsened = qt.apply_clearance_coarsening()

    # Must contain 1.0m super-tiles
    super_tiles = [l for l in coarsened if np.isclose(l.size, 1.0)]
    assert len(super_tiles) >= 1, f"Expected 1.0m super-tiles, got {len(super_tiles)}"
    for st in super_tiles:
        assert st.cost == 0, f"Expected super-tile cost=0, got {st.cost}"
        assert st.is_leaf is True
        assert st.stats is not None
        assert np.isclose(st.stats.mean_z, -1.60, atol=0.05)


def test_constricted_pinch_point_scales_to_075m_caution_tiles():
    """
    Verifies that a narrow corridor with clearance between 0.6m and 1.0m
    (pinch-point with total width ~1.2m to 1.8m) generates 0.75m x 0.75m
    Caution tiles (Cost = 100).
    """
    qt = AdaptiveQuadtree()
    x_min = qt.bounds.X_MIN
    y_min = qt.bounds.Y_MIN
    res = qt.coarse_res

    # Create a corridor with a protruding pinch obstacle (width constricted to ~1.5m)
    # Producing diagonal clearance = sqrt(2)*0.5m = 0.707m (falls in [0.6m, 1.0m))
    leaves = []
    for gx in range(10, 16):
        for gy in range(10, 14):
            # Protruding obstacle corners pinching the corridor
            if (gx == 12 and gy == 10) or (gx == 13 and gy == 13):
                leaves.append(QuadtreeNode(
                    x=x_min + (gx + 0.5) * res,
                    y=y_min + (gy + 0.5) * res,
                    size=res,
                    depth=0,
                    cost=255,
                    is_leaf=True,
                    stats=CellStats(-0.5, 0.5, 1.0, 0.1, 0.0, 10, 0.0),
                ))
            else:
                leaves.append(QuadtreeNode(
                    x=x_min + (gx + 0.5) * res,
                    y=y_min + (gy + 0.5) * res,
                    size=res,
                    depth=0,
                    cost=0,
                    is_leaf=True,
                    stats=CellStats(-1.62, -1.58, 0.04, 0.0001, 0.0, 10, -1.60),
                ))

    qt.leaves = leaves
    coarsened = qt.apply_clearance_coarsening()

    # The ground corridor must contain 0.75m Caution tiles
    caution_tiles = [l for l in coarsened if np.isclose(l.size, 0.75)]
    assert len(caution_tiles) >= 1, f"Expected 0.75m pinch-point caution tiles, got {len(caution_tiles)}"
    for ct in caution_tiles:
        assert ct.cost == 100, f"Expected pinch-point cost=100, got {ct.cost}"
        assert ct.is_leaf is True


def test_obstacles_maintain_050m_coarse_leaves_cost_255():
    """
    Verifies that obstacles / lethal boundaries maintain consistent
    0.50m x 0.50m coarse leaves with cost = 255.
    """
    qt = AdaptiveQuadtree()
    obs_node = QuadtreeNode(
        x=10.25,
        y=2.25,
        size=0.50,
        depth=0,
        cost=255,
        is_leaf=True,
        stats=CellStats(-0.5, 0.5, 1.0, 0.1, 0.0, 20, 0.0),
    )
    qt.leaves = [obs_node]
    coarsened = qt.apply_clearance_coarsening()

    assert len(coarsened) == 1
    leaf = coarsened[0]
    assert np.isclose(leaf.size, 0.50)
    assert leaf.cost == 255
    assert leaf.is_leaf is True


def test_obstacle_closing_does_not_bleed_into_drivable_corridor():
    """
    Verifies that interpolate_obstacle_clusters strictly respects ~ground_2d
    and clearance > 0.6m so red blocks never bleed into drivable paths.
    """
    qt = AdaptiveQuadtree()
    x_min = qt.bounds.X_MIN
    y_min = qt.bounds.Y_MIN
    res = qt.coarse_res

    # Create ground cells across a wide road (gy from 10 to 20, gx from 10 to 20)
    leaves = []
    for gy in range(10, 21):
        for gx in range(10, 21):
            cx = x_min + (gx + 0.5) * res
            cy = y_min + (gy + 0.5) * res
            leaves.append(QuadtreeNode(
                x=cx,
                y=cy,
                size=res,
                depth=0,
                cost=0,
                is_leaf=True,
                stats=CellStats(-1.62, -1.58, 0.04, 0.0001, 0.0, 10, -1.60),
            ))

    # Add two obstacles on opposite sides of the road
    # Obstacle 1 at (gx=15, gy=9), Obstacle 2 at (gx=15, gy=21)
    obs1 = QuadtreeNode(
        x=x_min + (15 + 0.5) * res,
        y=y_min + (9 + 0.5) * res,
        size=res,
        depth=0,
        cost=255,
        is_leaf=True,
        stats=CellStats(-0.5, 0.5, 1.0, 0.1, 0.0, 10, 0.0),
    )
    obs2 = QuadtreeNode(
        x=x_min + (15 + 0.5) * res,
        y=y_min + (21 + 0.5) * res,
        size=res,
        depth=0,
        cost=255,
        is_leaf=True,
        stats=CellStats(-0.5, 0.5, 1.0, 0.1, 0.0, 10, 0.0),
    )
    leaves.extend([obs1, obs2])

    qt.leaves = leaves
    filled = qt.interpolate_obstacle_clusters()

    # None of the road cells (gy in [10, 20]) should have been converted to cost=255
    road_obstacles = [
        l for l in filled
        if l.cost == 255 and (10 <= int((l.y - y_min) / res) <= 20)
    ]
    assert len(road_obstacles) == 0, f"Obstacles bled into drivable corridor: {len(road_obstacles)} red cells found on road"
