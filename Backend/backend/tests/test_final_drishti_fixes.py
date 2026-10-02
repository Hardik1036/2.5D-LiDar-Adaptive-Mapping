"""Regression tests for Drishti 2.5D Adaptive Mapping Final Fixes.

Covers all 15 required verification items:
1.  10 m boundary (9.99 m -> 5 cm, 10.00 m -> 10 cm, 10.01 m -> 10 cm).
2.  25 m boundary (24.99 m -> 10 cm, 25.00 m -> 25 cm, 25.01 m -> 25 cm).
3.  50 m boundary (49.99 m -> 25 cm, 50.00 m -> 50 cm, 50.01 m -> 50 cm).
4.  100 m boundary (99.99 m -> 50 cm, 100.00 m -> 50 cm, >100.00 m -> outside perception envelope).
5.  Point exactly at each boundary (10.00 m, 25.00 m, 50.00 m, 100.00 m).
6.  Point just inside each boundary (9.99 m, 24.99 m, 49.99 m, 99.99 m).
7.  Point just outside each boundary (10.01 m, 25.01 m, 50.01 m, 100.01 m).
8.  Empty/no-return spatial cells (observed=False, valid=False, cost=-1).
9.  Unknown cells never receive fabricated elevation (NaN or None, never 0 or -1.68).
10. Unknown cells are distinguishable from cells outside perception envelope.
11. POINTPILLARS_ENABLED=false status contract.
12. POINTPILLARS_ENABLED=true status contract.
13. Missing model status (MODEL NOT LOADED).
14. Failed model initialisation status (FAILED, never converted to DISABLED).
15. Model validation telemetry fields (NuScenes val, mAP: 9.73%, NDS: 14.90%, GT: NOT AVAILABLE).
"""

import math
import numpy as np
import pytest

from backend.config import FOVEATED, MapBounds
from backend.mapping.foveated_grid import (
    FoveatedCell,
    FoveatedGrid,
    get_zone_for_distance,
    get_mandated_resolution,
)
from backend.mapping.quadtree import QuadtreeNode
from backend.server.payload_builder import PayloadBuilder, _serialize_leaf
from backend.adapters.ml_adapter import MLAdapter, ModelStatus


# ============================================================================
# TESTS 1 - 7: FOVEATED BOUNDARY RESOLUTION & DETERMINISM
# ============================================================================

class TestFoveatedBoundaries:
    """Tests 1 through 7: Exact radial boundary transitions and deterministic ownership."""

    def test_item_1_10m_boundary(self):
        """1. 10 m boundary: 9.99m -> 5cm, 10.00m -> 10cm, 10.01m -> 10cm."""
        assert get_zone_for_distance(9.99) == 1
        assert get_mandated_resolution(1) == pytest.approx(0.05)

        assert get_zone_for_distance(10.00) == 2
        assert get_mandated_resolution(2) == pytest.approx(0.10)

        assert get_zone_for_distance(10.01) == 2
        assert get_mandated_resolution(2) == pytest.approx(0.10)

    def test_item_2_25m_boundary(self):
        """2. 25 m boundary: 24.99m -> 10cm, 25.00m -> 25cm, 25.01m -> 25cm."""
        assert get_zone_for_distance(24.99) == 2
        assert get_mandated_resolution(2) == pytest.approx(0.10)

        assert get_zone_for_distance(25.00) == 3
        assert get_mandated_resolution(3) == pytest.approx(0.25)

        assert get_zone_for_distance(25.01) == 3
        assert get_mandated_resolution(3) == pytest.approx(0.25)

    def test_item_3_50m_boundary(self):
        """3. 50 m boundary: 49.99m -> 25cm, 50.00m -> 50cm, 50.01m -> 50cm."""
        assert get_zone_for_distance(49.99) == 3
        assert get_mandated_resolution(3) == pytest.approx(0.25)

        assert get_zone_for_distance(50.00) == 4
        assert get_mandated_resolution(4) == pytest.approx(0.50)

        assert get_zone_for_distance(50.01) == 4
        assert get_mandated_resolution(4) == pytest.approx(0.50)

    def test_item_4_100m_boundary(self):
        """4. 100 m boundary: 99.99m -> 50cm, 100.00m -> 50cm, >100.00m -> outside envelope."""
        assert get_zone_for_distance(99.99) == 4
        assert get_mandated_resolution(4) == pytest.approx(0.50)

        assert get_zone_for_distance(100.00) == 4
        assert get_mandated_resolution(4) == pytest.approx(0.50)

        # > 100.00m is zone 5 (outside perception envelope)
        assert get_zone_for_distance(100.01) == 5

    def test_item_5_points_exactly_at_boundaries(self):
        """5. Points placed exactly at 10.00, 25.00, 50.00, 100.00."""
        grid = FoveatedGrid(config=FOVEATED, bounds=MapBounds)

        # 10.00m boundary point
        p_10 = np.array([[10.00, 0.0, -1.6, 0.5]], dtype=np.float32)
        cells_10 = grid.build_foveated(p_10)
        assert len(cells_10) == 1
        assert cells_10[0].size == pytest.approx(0.10)
        assert cells_10[0].zone == 2

        # 25.00m boundary point
        p_25 = np.array([[25.00, 0.0, -1.6, 0.5]], dtype=np.float32)
        cells_25 = grid.build_foveated(p_25)
        assert len(cells_25) == 1
        assert cells_25[0].size == pytest.approx(0.25)
        assert cells_25[0].zone == 3

        # 50.00m boundary point
        p_50 = np.array([[50.00, 0.0, -1.6, 0.5]], dtype=np.float32)
        cells_50 = grid.build_foveated(p_50)
        assert len(cells_50) == 1
        assert cells_50[0].size == pytest.approx(0.50)
        assert cells_50[0].zone == 4

        # 100.00m boundary point
        p_100 = np.array([[100.00, 0.0, -1.6, 0.5]], dtype=np.float32)
        cells_100 = grid.build_foveated(p_100)
        assert len(cells_100) == 1
        assert cells_100[0].size == pytest.approx(0.50)
        assert cells_100[0].zone == 4

    def test_item_6_points_just_inside_boundaries(self):
        """6. Points placed just inside each boundary (9.99m, 24.99m, 49.99m, 99.99m)."""
        grid = FoveatedGrid(config=FOVEATED, bounds=MapBounds)
        pts = np.array([
            [9.99, 0.0, -1.6, 0.5],
            [24.99, 0.0, -1.6, 0.5],
            [49.99, 0.0, -1.6, 0.5],
            [99.99, 0.0, -1.6, 0.5],
        ], dtype=np.float32)
        cells = grid.build_foveated(pts)
        res_map = {c.zone: c.size for c in cells}
        assert res_map[1] == pytest.approx(0.05)
        assert res_map[2] == pytest.approx(0.10)
        assert res_map[3] == pytest.approx(0.25)
        assert res_map[4] == pytest.approx(0.50)

    def test_item_7_points_just_outside_boundaries(self):
        """7. Points placed just outside each boundary (10.01m, 25.01m, 50.01m, 100.01m)."""
        grid = FoveatedGrid(config=FOVEATED, bounds=MapBounds)
        pts = np.array([
            [10.01, 0.0, -1.6, 0.5],
            [25.01, 0.0, -1.6, 0.5],
            [50.01, 0.0, -1.6, 0.5],
        ], dtype=np.float32)
        cells = grid.build_foveated(pts)
        res_map = {c.zone: c.size for c in cells}
        assert res_map[2] == pytest.approx(0.10)
        assert res_map[3] == pytest.approx(0.25)
        assert res_map[4] == pytest.approx(0.50)

        # > 100.00m point outside radial envelope
        p_out = np.array([[100.01, 0.0, -1.6, 0.5]], dtype=np.float32)
        cells_out = grid.build_foveated(p_out)
        assert len(cells_out) == 0  # Excluded from perception envelope

    def test_zero_spatial_overlaps_across_dense_sweep(self):
        """Verify zero spatial overlap across all zones simultaneously."""
        grid = FoveatedGrid(config=FOVEATED, bounds=MapBounds)
        r_vals = np.linspace(0.5, 95.0, 500)
        angles = np.linspace(-np.pi / 4, np.pi / 4, 500)
        xs = r_vals * np.cos(angles)
        ys = r_vals * np.sin(angles)
        zs = np.full_like(xs, -1.6)
        intensities = np.full_like(xs, 0.5)
        pts = np.column_stack([xs, ys, zs, intensities]).astype(np.float32)

        cells = grid.build_foveated(pts)
        assert len(cells) > 0

        # Check for overlaps via 5cm base spatial hash claims
        claimed_bins = set()
        overlap_found = False
        for c in cells:
            ix_min = int(round((c.x_min - MapBounds.X_MIN) / 0.05))
            ix_max = int(round((c.x_max - MapBounds.X_MIN) / 0.05))
            iy_min = int(round((c.y_min - MapBounds.Y_MIN) / 0.05))
            iy_max = int(round((c.y_max - MapBounds.Y_MIN) / 0.05))
            for ix in range(ix_min, ix_max):
                for iy in range(iy_min, iy_max):
                    key = (ix, iy)
                    if key in claimed_bins:
                        overlap_found = True
                        break
                    claimed_bins.add(key)
                if overlap_found:
                    break
            if overlap_found:
                break

        assert not overlap_found, "Detected overlapping spatial cells in foveated representation!"


# ============================================================================
# TESTS 8 - 10: UNOBSERVED CELLS & SEPARATION OF COVERAGE VS OBSERVATION
# ============================================================================

class TestUnobservedCellSemantics:
    """Tests 8 through 10: Explicit validity state and no fabricated terrain."""

    def test_item_8_empty_no_return_cell_state(self):
        """8. Empty/no-return spatial cells have observed=False, valid=False, cost=-1."""
        cell = FoveatedCell.create_unobserved_cell(x=5.0, y=5.0, size=0.05, zone=1)
        assert cell.observed is False
        assert cell.valid is False
        assert cell.point_count == 0
        assert cell.cost == -1.0

        d = cell.to_dict()
        assert d["observed"] is False
        assert d["valid"] is False
        assert d["cost"] == -1.0

    def test_item_9_unknown_cells_never_receive_fabricated_elevation(self):
        """9. Unknown cells never receive fabricated elevation (NaN for getters, None for JSON)."""
        cell = FoveatedCell.create_unobserved_cell(x=2.0, y=2.0, size=0.05, zone=1)
        assert math.isnan(cell.z_min)
        assert math.isnan(cell.z_max)
        assert math.isnan(cell.mean_z)
        assert math.isnan(cell.variance)
        assert math.isnan(cell.delta_z)
        assert math.isnan(cell.slope)

        # QuadtreeNode unobserved representation
        node = QuadtreeNode(x=2.0, y=2.0, size=0.5, depth=4, observed=False, valid=False)
        assert math.isnan(node.z_min)
        assert math.isnan(node.z_max)
        assert math.isnan(node.mean_z)
        assert math.isnan(node.variance)
        assert math.isnan(node.delta_z)
        assert math.isnan(node.slope)

        # Serialized payload format: None (JSON-compliant null), NEVER 0.0 or -1.68
        leaf_dict = _serialize_leaf(node)
        assert leaf_dict["observed"] is False
        assert leaf_dict["valid"] is False
        assert leaf_dict["mean_z"] is None
        assert leaf_dict["z_min"] is None
        assert leaf_dict["z_max"] is None
        assert leaf_dict["variance"] is None
        assert leaf_dict["delta_z"] is None
        assert leaf_dict["slope"] is None
        assert leaf_dict["cost"] == -1.0

    def test_item_10_unknown_distinguishable_from_outside_envelope(self):
        """10. Unknown cells inside coverage are distinguishable from points outside envelope."""
        # Unobserved cell inside coverage (r = 5.0m < 100m)
        unobs_cell = FoveatedCell.create_unobserved_cell(x=5.0, y=0.0, size=0.05, zone=1)
        assert unobs_cell.observed is False
        assert unobs_cell.valid is False
        assert unobs_cell.zone == 1  # Inside perception envelope zone 1

        # Point outside perception envelope (> 100.0m)
        grid = FoveatedGrid(config=FOVEATED, bounds=MapBounds)
        p_out = np.array([[105.0, 0.0, -1.6, 0.5]], dtype=np.float32)
        cells_out = grid.build_foveated(p_out)
        # Outside envelope generates NO cells at all
        assert len(cells_out) == 0


# ============================================================================
# TESTS 11 - 14: PERCEPTION ENGINE STATUS PANEL
# ============================================================================

class TestPerceptionEngineStatus:
    """Tests 11 through 14: Decoupled status system contracts."""

    def test_item_11_pointpillars_enabled_false_status(self):
        """11. POINTPILLARS_ENABLED=false contract:
        Mode: 2.5D ADAPTIVE
        Segmentation: GEOMETRIC
        3D Detection: OPTIONAL / DISABLED
        Ground Truth: NOT AVAILABLE
        PointPillars: AVAILABLE / DISABLED
        """
        adapter = MLAdapter()
        status = adapter.get_model_status(pointpillars_enabled=False)

        assert status["perception_engine"]["mode"] == "2.5D ADAPTIVE"
        assert status["perception_engine"]["segmentation"] == "GEOMETRIC"
        assert status["perception_engine"]["detection_3d"] == "OPTIONAL / DISABLED"
        assert status["perception_engine"]["ground_truth"] == "NOT AVAILABLE"
        assert status["models"]["pointpillars"] == "AVAILABLE / DISABLED"

    def test_item_12_pointpillars_enabled_true_status(self):
        """12. POINTPILLARS_ENABLED=true contract (when loaded):
        Mode: LIVE POINTPILLARS
        Segmentation: GEOMETRIC
        3D Detection: POINTPILLARS ACTIVE
        Ground Truth: NOT AVAILABLE
        PointPillars: ACTIVE
        """
        adapter = MLAdapter()
        # Mock active loaded state for testing status rendering contract
        adapter._pp_status = ModelStatus.ACTIVE
        status = adapter.get_model_status(pointpillars_enabled=True)

        assert status["perception_engine"]["mode"] == "LIVE POINTPILLARS"
        assert status["perception_engine"]["segmentation"] == "GEOMETRIC"
        assert status["perception_engine"]["detection_3d"] == "POINTPILLARS ACTIVE"
        assert status["perception_engine"]["ground_truth"] == "NOT AVAILABLE"
        assert status["models"]["pointpillars"] == "ACTIVE"

    def test_item_13_missing_model_status(self):
        """13. Missing model status: MODEL NOT LOADED."""
        adapter = MLAdapter()
        adapter._pp_status = ModelStatus.NOT_LOADED
        status = adapter.get_model_status(pointpillars_enabled=True)

        assert status["models"]["pointpillars"] == "MODEL NOT LOADED"

    def test_item_14_failed_model_initialisation_status(self):
        """14. Failed model initialisation: FAILED (never silently converted to DISABLED)."""
        adapter = MLAdapter()
        adapter._pp_status = ModelStatus.FAILED
        status = adapter.get_model_status(pointpillars_enabled=True)

        assert status["models"]["pointpillars"] == "FAILED"
        assert status["models"]["pointpillars"] != "AVAILABLE / DISABLED"
        assert status["perception_engine"]["detection_3d"] == "INITIALISATION FAILED"


# ============================================================================
# TEST 15: MODEL VALIDATION TELEMETRY FIELDS
# ============================================================================

class TestModelValidationTelemetry:
    """Test 15: PointPillars model validation metrics block."""

    def test_item_15_model_validation_telemetry(self):
        """15. Model validation telemetry exposed with exact NuScenes metrics,
        and live ground truth explicitly NOT AVAILABLE.
        """
        adapter = MLAdapter()
        status = adapter.get_model_status(pointpillars_enabled=False)
        val = status["model_validation"]

        assert val["dataset"] == "NuScenes validation"
        assert val["samples"] == 81
        assert val["map"] == pytest.approx(0.0973)
        assert val["nds"] == pytest.approx(0.1490)
        assert val["map_pct"] == "9.73%"
        assert val["nds_pct"] == "14.90%"
        assert val["per_class_ap"]["car"] == "41.9%"
        assert val["per_class_ap"]["pedestrian"] == "40.3%"
        assert val["per_class_ap"]["truck"] == "7.0%"
        assert val["per_class_ap"]["bus"] == "8.1%"
        assert val["live_ground_truth"] == "NOT AVAILABLE"

        # Verify full payload builder includes model_validation and honest ground truth
        builder = PayloadBuilder()
        payload = builder.build_payload(
            frame_id=1,
            timestamp=0.0,
            system_stats={"point_count": 100, "ground_truth_status": "NOT AVAILABLE"},
            leaves=[],
            tracks=[],
            hazard_cones={},
            model_status=status,
        )
        assert "model_validation" in payload
        payload_dict = payload._dict
        assert payload_dict["model_validation"]["mAP"] == pytest.approx(9.73)
        assert payload_dict["model_validation"]["NDS"] == pytest.approx(14.90)
        assert payload_dict["system_stats"]["ground_truth_status"] == "NOT AVAILABLE"
        assert payload_dict["mode"] == "2.5D ADAPTIVE"
