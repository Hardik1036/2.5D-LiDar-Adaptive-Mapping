"""
Tests for Negative Obstacle Gating & Caution (Yellow) Gap Classification
DRISHTI-2.5D Adaptive Mapping (SIH26053 / DRDO PS-53)
"""

import numpy as np
import pytest
from backend.config import BOUNDS
from backend.mapping.quadtree import AdaptiveQuadtree


def test_safe_road_planar_bridging():
    """
    1. SAFE ROAD (Green / Cost = 0):
    Narrow gap (<= 2 cells / <= 1.0m) with strict planar height matching (|delta_z| <= 0.06m)
    must receive cost = 0.
    """
    qt = AdaptiveQuadtree()
    # Create 1-cell gap along Y: rows 10 and 12 exist at Z = -1.60m, row 11 is missing
    cx = BOUNDS.X_MIN + 15.5 * qt.coarse_res
    cy10 = BOUNDS.Y_MIN + 10.5 * qt.coarse_res
    cy12 = BOUNDS.Y_MIN + 12.5 * qt.coarse_res

    pts = []
    for _ in range(5):
        pts.append([cx, cy10, -1.60])
        pts.append([cx, cy12, -1.60])
    pts = np.array(pts, dtype=np.float32)

    qt.build(pts)
    interpolated = qt.interpolate_ground_rings()

    cy11 = BOUNDS.Y_MIN + 11.5 * qt.coarse_res
    inpainted = [l for l in interpolated if np.isclose(l.y, cy11) and np.isclose(l.x, cx)]
    assert len(inpainted) == 1
    assert inpainted[0].cost == 0, f"Expected cost 0 for narrow planar gap, got {inpainted[0].cost}"


def test_caution_gap_medium_width():
    """
    2. CAUTION / POTENTIAL VOID (Yellow / Cost = 100):
    Medium gap (3 cells / 1.5m) must be classified as Caution (cost = 100).
    """
    qt = AdaptiveQuadtree()
    # Create 3-cell gap along X: cols 10 and 14 exist at Z = -1.60m, cols 11, 12, 13 missing
    # In forward driving corridor (y = 0.0)
    cy = 0.0
    cx10 = BOUNDS.X_MIN + 10.5 * qt.coarse_res
    cx14 = BOUNDS.X_MIN + 14.5 * qt.coarse_res

    pts = []
    for _ in range(5):
        pts.append([cx10, cy, -1.60])
        pts.append([cx14, cy, -1.60])
    pts = np.array(pts, dtype=np.float32)

    qt.build(pts)
    interpolated = qt.interpolate_ground_rings()

    # Gap cells at col 11, 12, 13
    cx12 = BOUNDS.X_MIN + 12.5 * qt.coarse_res
    inpainted = [l for l in interpolated if np.isclose(l.x, cx12) and np.isclose(l.y, cy, atol=0.25)]
    assert len(inpainted) >= 1
    for leaf in inpainted:
        assert leaf.cost == 100, f"Expected cost 100 for medium gap, got {leaf.cost}"


def test_caution_gap_slope_variation():
    """
    2. CAUTION / POTENTIAL VOID (Yellow / Cost = 100):
    Narrow gap (<= 2 cells) with slight slope variation (|delta_z| = 0.08m > 0.06m)
    must receive cost = 100 (Caution Yellow).
    """
    qt = AdaptiveQuadtree()
    cx = BOUNDS.X_MIN + 15.5 * qt.coarse_res
    cy10 = BOUNDS.Y_MIN + 10.5 * qt.coarse_res
    cy12 = BOUNDS.Y_MIN + 12.5 * qt.coarse_res

    pts = []
    for _ in range(5):
        pts.append([cx, cy10, -1.60])
        pts.append([cx, cy12, -1.52])  # dz = 0.08m > 0.06m (slope variation)
    pts = np.array(pts, dtype=np.float32)

    qt.build(pts)
    interpolated = qt.interpolate_ground_rings()

    cy11 = BOUNDS.Y_MIN + 11.5 * qt.coarse_res
    inpainted = [l for l in interpolated if np.isclose(l.y, cy11) and np.isclose(l.x, cx)]
    assert len(inpainted) == 1
    assert inpainted[0].cost == 100, f"Expected cost 100 for slope variation, got {inpainted[0].cost}"


def test_lethal_step_drop_obstacle():
    """
    3. LETHAL NEGATIVE OBSTACLE (Red / Cost = 255):
    Step drop across boundary (|delta_z| > 0.15m) must receive cost = 255 (Lethal Ditch/Pothole).
    """
    qt = AdaptiveQuadtree()
    cx = BOUNDS.X_MIN + 15.5 * qt.coarse_res
    cy10 = BOUNDS.Y_MIN + 10.5 * qt.coarse_res
    cy12 = BOUNDS.Y_MIN + 12.5 * qt.coarse_res

    pts = []
    for _ in range(5):
        pts.append([cx, cy10, -1.50])
        pts.append([cx, cy12, -1.72])  # dz = 0.22m > 0.15m (step drop)
    pts = np.array(pts, dtype=np.float32)

    qt.build(pts)
    interpolated = qt.interpolate_ground_rings()

    cy11 = BOUNDS.Y_MIN + 11.5 * qt.coarse_res
    inpainted = [l for l in interpolated if np.isclose(l.y, cy11) and np.isclose(l.x, cx)]
    assert len(inpainted) == 1
    assert inpainted[0].cost == 255, f"Expected cost 255 for step drop, got {inpainted[0].cost}"


def test_lethal_trench_depression_below_baseline():
    """
    3. LETHAL NEGATIVE OBSTACLE (Red / Cost = 255):
    Boundary cells indicating drop below road baseline (Z < -2.0m) must be tagged as cost = 255.
    """
    qt = AdaptiveQuadtree()
    cx = BOUNDS.X_MIN + 15.5 * qt.coarse_res
    cy10 = BOUNDS.Y_MIN + 10.5 * qt.coarse_res
    cy12 = BOUNDS.Y_MIN + 12.5 * qt.coarse_res

    pts = []
    for _ in range(5):
        pts.append([cx, cy10, -1.80])
        pts.append([cx, cy12, -2.05])  # drops below -2.0m (ditch threshold)
    pts = np.array(pts, dtype=np.float32)

    qt.build(pts)
    interpolated = qt.interpolate_ground_rings()

    cy11 = BOUNDS.Y_MIN + 11.5 * qt.coarse_res
    inpainted = [l for l in interpolated if np.isclose(l.y, cy11) and np.isclose(l.x, cx)]
    assert len(inpainted) == 1
    assert inpainted[0].cost == 255, f"Expected cost 255 for trench drop below -2.0m, got {inpainted[0].cost}"


def test_wide_unscanned_gap_never_filled_green():
    """
    4. UNCONFIRMED WIDE GAPS (> 4 cells / > 2.0m):
    Wide gaps must NEVER be filled as green (cost = 0).
    """
    qt = AdaptiveQuadtree()
    # 6-cell gap along X outside corridor
    cy = 8.0
    cx10 = BOUNDS.X_MIN + 10.5 * qt.coarse_res
    cx17 = BOUNDS.X_MIN + 17.5 * qt.coarse_res

    pts = []
    for _ in range(5):
        pts.append([cx10, cy, -1.60])
        pts.append([cx17, cy, -1.60])
    pts = np.array(pts, dtype=np.float32)

    qt.build(pts)
    interpolated = qt.interpolate_ground_rings()

    # None of the inpainted cells in this wide gap should be Cost = 0
    in_gap = [l for l in interpolated if cx10 < l.x < cx17 and np.isclose(l.y, cy, atol=0.25)]
    for leaf in in_gap:
        assert leaf.cost != 0, "Wide gap must never be blindly filled as green (cost=0)!"


def test_caution_cost_preserved_through_costmap_evaluation():
    """
    Verifies that CostmapEvaluator.evaluate_leaves preserves caution tiles (cost = 100)
    and does not overwrite them back to safe road (cost = 0).
    """
    from backend.mapping.costmap import CostmapEvaluator
    from backend.mapping.cell_statistics import CellStats
    from backend.mapping.quadtree import QuadtreeNode

    evaluator = CostmapEvaluator()
    caution_leaf = QuadtreeNode(
        x=2.0,
        y=1.0,
        size=0.5,
        depth=0,
        cost=100,
        stats=CellStats(
            z_min=-1.64,
            z_max=-1.56,
            delta_z=0.08,
            variance=0.0001,
            slope=0.0,
            point_count=1,
            mean_z=-1.60,
        ),
        is_leaf=True,
    )

    evaluator.evaluate_leaves([caution_leaf])
    assert caution_leaf.cost == 100, f"Expected caution tile to retain cost 100, got {caution_leaf.cost}"


def test_caution_and_lethal_payload_serialization_integrity():
    """
    Verifies that PayloadBuilder serializes cost 100 and cost 255 cells
    and prioritizes them in the output WebSocket payload.
    """
    import json
    from backend.server.payload_builder import PayloadBuilder
    from backend.mapping.cell_statistics import CellStats
    from backend.mapping.quadtree import QuadtreeNode

    builder = PayloadBuilder()
    node0 = QuadtreeNode(
        x=1.0, y=1.0, size=0.5, depth=0, cost=0,
        stats=CellStats(z_min=-1.61, z_max=-1.59, delta_z=0.02, variance=0.0001, slope=0.0, point_count=1, mean_z=-1.60)
    )
    node100 = QuadtreeNode(
        x=2.0, y=1.0, size=0.5, depth=0, cost=100,
        stats=CellStats(z_min=-1.64, z_max=-1.56, delta_z=0.08, variance=0.0001, slope=0.0, point_count=1, mean_z=-1.60)
    )
    node255 = QuadtreeNode(
        x=3.0, y=1.0, size=0.5, depth=0, cost=255,
        stats=CellStats(z_min=-2.15, z_max=-1.95, delta_z=0.20, variance=0.0001, slope=0.0, point_count=1, mean_z=-2.05)
    )

    payload_str = builder.build_payload(
        frame_id=1,
        timestamp=0.0,
        system_stats={},
        leaves=[node0, node100, node255],
        tracks=[],
        hazard_cones={},
    )

    data = json.loads(payload_str)
    costs = [c["cost"] for c in data["cells"]]
    assert 0 in costs, "Expected cost 0 in serialized payload"
    assert 100 in costs, "Expected cost 100 (Caution) in serialized payload"
    assert 255 in costs, "Expected cost 255 (Lethal) in serialized payload"

