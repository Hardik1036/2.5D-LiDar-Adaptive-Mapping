"""
Tests for Obstacle Coarse-Leaf Collapse (Coarse Tile Aggregation for Obstacle Clusters)
DRISHTI-2.5D Adaptive Mapping (SIH26053 / DRDO PS-53)
"""

import numpy as np
import pytest
from backend.config import BOUNDS
from backend.mapping.quadtree import AdaptiveQuadtree


def test_dense_obstacle_cluster_collapses_to_single_coarse_leaf():
    """
    Verifies that a dense obstacle cluster (>= 8 non-ground points with Z > -1.2m)
    collapses into a single unified coarse leaf (size = 0.50m, cost = 255)
    instead of fragmenting into 64 micro-cubes.
    """
    qt = AdaptiveQuadtree()
    # Create points strictly inside a single coarse cell [10.0, 10.5] x [2.0, 2.5]
    # Representing a solid obstacle (e.g. parked vehicle body, wall) at Z = -0.5m to 0.5m
    cx = 10.25
    cy = 2.25
    np.random.seed(42)
    obs_x = np.random.uniform(10.1, 10.4, 30)
    obs_y = np.random.uniform(2.1, 2.4, 30)
    obs_z = np.random.uniform(-0.8, 0.5, 30)  # All points well above -1.2m baseline

    pts = np.column_stack([obs_x, obs_y, obs_z])
    leaves = qt.build(pts)

    # Must contain exactly 1 leaf for this cell, NOT subdivided
    cell_leaves = [l for l in leaves if np.isclose(l.x, cx, atol=0.25) and np.isclose(l.y, cy, atol=0.25)]
    assert len(cell_leaves) == 1, f"Expected 1 collapsed coarse leaf, but got {len(cell_leaves)} leaves"

    leaf = cell_leaves[0]
    assert np.isclose(leaf.size, qt.coarse_res, atol=1e-3), f"Expected coarse size {qt.coarse_res}, got {leaf.size}"
    assert leaf.cost == 255, f"Expected lethal cost 255, got {leaf.cost}"
    assert leaf.is_leaf is True
    assert leaf.stats is not None
    assert leaf.stats.delta_z >= 0.25
    assert leaf.stats.z_max > -1.2


def test_boundary_transition_subdivides_into_fine_leaves():
    """
    Verifies that a cell containing a boundary transition (part ground, part obstacle/curb)
    genuinely subdivides into fine cells to preserve boundary fidelity.
    """
    qt = AdaptiveQuadtree()
    # Boundary cell [15.0, 15.5] x [0.0, 0.5]
    # Ground half: X in [15.05, 15.20], Z = -1.60m
    # Curb/obstacle half: X in [15.30, 15.45], Z = -1.35m (step height = 0.25m)
    np.random.seed(42)
    gnd_x = np.random.uniform(15.05, 15.20, 20)
    gnd_y = np.random.uniform(0.1, 0.4, 20)
    gnd_z = np.full(20, -1.60)

    curb_x = np.random.uniform(15.30, 15.45, 20)
    curb_y = np.random.uniform(0.1, 0.4, 20)
    curb_z = np.full(20, -1.35)

    pts = np.column_stack([
        np.concatenate([gnd_x, curb_x]),
        np.concatenate([gnd_y, curb_y]),
        np.concatenate([gnd_z, curb_z]),
    ])

    leaves = qt.build(pts)
    cell_leaves = [l for l in leaves if 15.0 <= l.x <= 15.5 and 0.0 <= l.y <= 0.5]
    # Boundary transition must subdivide into multiple fine leaves
    assert len(cell_leaves) > 1, f"Boundary transition failed to subdivide (got {len(cell_leaves)} leaves)"
    for l in cell_leaves:
        assert l.size < qt.coarse_res, f"Expected fine cell size < {qt.coarse_res}, got {l.size}"


def test_subdivide_recursive_coarse_obstacle_collapse():
    """
    Verifies that _subdivide_recursive also respects coarse obstacle collapse for depth 0.
    """
    qt = AdaptiveQuadtree()
    np.random.seed(42)
    obs_pts = np.column_stack([
        np.random.uniform(1.1, 1.4, 25),
        np.random.uniform(1.1, 1.4, 25),
        np.random.uniform(-0.5, 0.8, 25),
    ])

    node = qt._subdivide_recursive(cx=1.25, cy=1.25, size=qt.coarse_res, depth=0, pts=obs_pts)
    assert node.is_leaf is True
    assert node.cost == 255
    assert node.size == qt.coarse_res


def test_interpolate_obstacle_clusters_bypassed():
    """
    Verifies that interpolate_obstacle_clusters() is bypassed and leaves target leaves
    unchanged without artificially filling or dilating void spaces between real obstacles.
    """
    from backend.mapping.quadtree import QuadtreeNode
    from backend.mapping.cell_statistics import CellStats

    qt = AdaptiveQuadtree()
    # Create two obstacle cells separated by a 1-cell void at (10.25, 2.25)
    st1 = CellStats(z_min=-0.8, z_max=0.5, delta_z=1.3, variance=0.1, slope=0.0, point_count=20, mean_z=-0.15)
    node1 = QuadtreeNode(x=10.25, y=1.75, size=0.50, depth=0, stats=st1, cost=255, is_leaf=True)

    st2 = CellStats(z_min=-0.8, z_max=0.5, delta_z=1.3, variance=0.1, slope=0.0, point_count=20, mean_z=-0.15)
    node2 = QuadtreeNode(x=10.25, y=2.75, size=0.50, depth=0, stats=st2, cost=255, is_leaf=True)

    qt.leaves = [node1, node2]
    result_leaves = qt.interpolate_obstacle_clusters()

    # Voids should NOT be bridged; only original leaves remain
    assert len(result_leaves) == 2
    bridged = [l for l in result_leaves if np.isclose(l.x, 10.25) and np.isclose(l.y, 2.25)]
    assert len(bridged) == 0, "No artificial obstacle leaves should be synthesized in voids"


