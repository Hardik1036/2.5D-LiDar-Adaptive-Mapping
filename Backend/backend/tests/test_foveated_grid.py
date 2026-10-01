import pytest
import numpy as np

from backend.mapping.foveated_grid import (
    FoveatedGrid,
    cell_footprint_r_min_r_max,
    get_zone_for_distance,
    get_mandated_resolution,
    check_zone_straddling,
    make_deterministic_cell_id,
)
from backend.config import FOVEATED, MapBounds


class TestRadialBoundaries:
    """12.1 Radial Boundary Tests."""

    @pytest.mark.parametrize(
        "dist, expected_zone, expected_res",
        [
            (9.99, 1, 0.05),
            (10.00, 2, 0.10),
            (10.01, 2, 0.10),
            (24.99, 2, 0.10),
            (25.00, 3, 0.25),
            (25.01, 3, 0.25),
            (49.99, 3, 0.25),
            (50.00, 4, 0.50),
            (50.01, 4, 0.50),
            (99.99, 4, 0.50),
            (100.00, 4, 0.50),
        ],
    )
    def test_radial_boundary_zone_assignment(self, dist, expected_zone, expected_res):
        zone = get_zone_for_distance(dist)
        res = get_mandated_resolution(zone)
        assert zone == expected_zone, f"At {dist}m expected zone {expected_zone}, got {zone}"
        assert abs(res - expected_res) < 1e-6, f"At {dist}m expected res {expected_res}, got {res}"

    def test_boundary_points_produce_valid_cell_ids(self):
        distances = [9.99, 10.00, 10.01, 24.99, 25.00, 25.01, 49.99, 50.00, 50.01, 99.99, 100.00]
        grid = FoveatedGrid(config=FOVEATED, bounds=MapBounds)

        pts = []
        for d in distances:
            pts.append([d, 0.0, -1.6, 0.5])
            pts.append([0.0, min(d, 49.9), -1.6, 0.5])
            pts.append([-min(d, 19.9), 0.0, -1.6, 0.5])

        points_arr = np.array(pts, dtype=np.float32)
        cells = grid.build_foveated(points_arr)

        assert len(cells) > 0
        seen_ids = set()
        for c in cells:
            assert c.cell_id is not None
            assert len(c.cell_id) > 0
            assert c.cell_id not in seen_ids, f"Duplicate cell ID: {c.cell_id}"
            seen_ids.add(c.cell_id)


class TestExactCellResolution:
    """12.2 Exact Cell Resolution."""

    def test_exact_cell_resolutions(self):
        grid = FoveatedGrid(config=FOVEATED, bounds=MapBounds)
        # Create points across all 4 zones
        pts = np.array([
            [5.0, 2.0, -1.6, 0.5],    # Zone 1 (0-10m)
            [15.0, 5.0, -1.6, 0.5],   # Zone 2 (10-25m)
            [35.0, 10.0, -1.6, 0.5],  # Zone 3 (25-50m)
            [75.0, 20.0, -1.6, 0.5],  # Zone 4 (50-100m)
        ], dtype=np.float32)

        cells = grid.build_foveated(pts)
        for c in cells:
            dist = np.hypot(c.x, c.y)
            if dist < 10.0:
                assert abs(c.size - 0.05) < 1e-5, f"Zone 1 leaf size must be 0.05m, got {c.size}"
            elif dist < 25.0:
                assert abs(c.size - 0.10) < 1e-5, f"Zone 2 leaf size must be 0.10m, got {c.size}"
            elif dist < 50.0:
                assert abs(c.size - 0.25) < 1e-5, f"Zone 3 leaf size must be 0.25m, got {c.size}"
            elif dist <= 100.0:
                assert abs(c.size - 0.50) < 1e-5, f"Zone 4 leaf size must be 0.50m, got {c.size}"

    def test_leaves_are_axis_aligned_rectangles(self):
        grid = FoveatedGrid(config=FOVEATED, bounds=MapBounds)
        pts = np.array([
            [4.0, 3.0, -1.6, 0.5],
            [12.0, 8.0, -1.6, 0.5],
            [30.0, 15.0, -1.6, 0.5],
            [60.0, 25.0, -1.6, 0.5],
        ], dtype=np.float32)
        cells = grid.build_foveated(pts)
        for c in cells:
            assert abs((c.x_max - c.x_min) - c.size) < 1e-5
            assert abs((c.y_max - c.y_min) - c.size) < 1e-5
            assert abs((c.x_min + c.x_max) * 0.5 - c.x) < 1e-5
            assert abs((c.y_min + c.y_max) * 0.5 - c.y) < 1e-5


class TestNoZoneStraddling:
    """12.3 No Zone-Straddling Tests."""

    def test_no_zone_straddling(self):
        grid = FoveatedGrid(config=FOVEATED, bounds=MapBounds)
        radii = np.linspace(1.0, 95.0, 300)
        angles = np.linspace(-np.pi / 3, np.pi / 3, 20)
        pts = []
        for r in radii:
            for a in angles:
                x = float(r * np.cos(a))
                y = float(r * np.sin(a))
                if MapBounds.X_MIN <= x <= MapBounds.X_MAX and MapBounds.Y_MIN <= y <= MapBounds.Y_MAX:
                    pts.append([x, y, -1.6, 0.5])

        points_arr = np.array(pts, dtype=np.float32)
        cells = grid.build_foveated(points_arr)

        for c in cells:
            r_min, r_max = cell_footprint_r_min_r_max(c.x_min, c.x_max, c.y_min, c.y_max)
            z_min_zone = get_zone_for_distance(r_min)
            z_max_zone = get_zone_for_distance(r_max)
            straddles = check_zone_straddling(r_min, r_max)
            assert not straddles, (
                f"Cell {c.cell_id} straddles zones! r_min={r_min:.4f}, r_max={r_max:.4f}, "
                f"z_min_zone={z_min_zone}, z_max_zone={z_max_zone}"
            )


class TestNoOverlappingCells:
    """12.4 No Overlapping Cells Tests."""

    def test_no_overlapping_cells(self):
        grid = FoveatedGrid(config=FOVEATED, bounds=MapBounds)
        pts = np.array([
            [2.0, 1.0, -1.6, 0.5],
            [2.02, 1.02, -1.6, 0.5],
            [12.5, 4.2, -1.6, 0.5],
            [12.55, 4.25, -1.6, 0.5],
            [30.0, -10.0, -1.6, 0.5],
            [60.0, 20.0, -1.6, 0.5],
        ], dtype=np.float32)

        cells = grid.build_foveated(pts)
        n = len(cells)
        for i in range(n):
            for j in range(i + 1, n):
                c1 = cells[i]
                c2 = cells[j]
                overlap_x = min(c1.x_max, c2.x_max) - max(c1.x_min, c2.x_min)
                overlap_y = min(c1.y_max, c2.y_max) - max(c1.y_min, c2.y_min)
                assert not (overlap_x > 1e-5 and overlap_y > 1e-5), (
                    f"Cells {c1.cell_id} and {c2.cell_id} overlap! "
                    f"overlap_x={overlap_x}, overlap_y={overlap_y}"
                )


class TestNoUncoveredCells:
    """12.5 No Uncovered Supported Regions Tests."""

    def test_no_uncovered_cells(self):
        grid = FoveatedGrid(config=FOVEATED, bounds=MapBounds)
        test_pts = [
            [3.0, 2.0],
            [12.0, -4.0],
            [35.0, 15.0],
            [70.0, -25.0],
        ]
        pts_arr = np.array([[p[0], p[1], -1.6, 0.5] for p in test_pts], dtype=np.float32)
        cells = grid.build_foveated(pts_arr)

        for px, py in test_pts:
            found = False
            for c in cells:
                if c.x_min - 1e-5 <= px <= c.x_max + 1e-5 and c.y_min - 1e-5 <= py <= c.y_max + 1e-5:
                    found = True
                    break
            assert found, f"Supported test point ({px}, {py}) was not covered by any leaf cell!"


class TestRadialEnvelopeSeparation:
    """
    PS 26053 Items 7 & 8: Radial Perception Envelope vs Rectangular Storage Envelope.
    Rectangular processing bounds: X: [-20, 100], Y: [-50, 50] (120m x 100m).
    Radial perception envelope: r <= 100.0 m.
    Expected coverage: rectangle ∩ radial perception envelope.
    """

    def test_outside_radial_envelope_not_zoned(self):
        """
        Points inside rectangular bounds but outside the 100m radial envelope
        (e.g., (90, 48), (95, 40), (100, 30)) must NOT be assigned to foveated
        perception zones (zones 1-4).
        """
        from backend.mapping.foveated_grid import is_in_radial_envelope

        # Points inside 120m x 100m rectangle, but with Euclidean distance > 100m
        test_points_outside = [
            (90.0, 48.0),   # r = sqrt(8100 + 2304) = sqrt(10404) = 102.0m > 100m
            (95.0, 40.0),   # r = sqrt(9025 + 1600) = sqrt(10625) ≈ 103.08m > 100m
            (100.0, 30.0),  # r = sqrt(10000 + 900) = sqrt(10900) ≈ 104.40m > 100m
            (98.0, 45.0),   # r = sqrt(9604 + 2025) = sqrt(11629) ≈ 107.84m > 100m
            (100.0, 50.0),  # r = sqrt(10000 + 2500) = sqrt(12500) ≈ 111.80m > 100m
        ]

        grid = FoveatedGrid(config=FOVEATED, bounds=MapBounds)

        for px, py in test_points_outside:
            r = float(np.sqrt(px * px + py * py))
            # Verify coordinates are inside rectangular bounds
            assert MapBounds.X_MIN <= px <= MapBounds.X_MAX, f"Point ({px}, {py}) outside X bounds"
            assert MapBounds.Y_MIN <= py <= MapBounds.Y_MAX, f"Point ({px}, {py}) outside Y bounds"
            # Verify distance is strictly > 100m
            assert r > 100.0, f"Point ({px}, {py}) distance {r:.2f}m not > 100m"

            # 1. Helper function verification
            assert not is_in_radial_envelope(px, py, max_radial_range=100.0)

            # 2. Zone lookup verification: must return 5 (outside envelope), NOT 1, 2, 3, or 4
            zone = get_zone_for_distance(r)
            assert zone == 5, f"Point at r={r:.2f}m was assigned zone {zone}, expected 5 (outside)"

        # Ingest these points into FoveatedGrid - they must be excluded from perception zones 1-4
        pts_arr = np.array([[p[0], p[1], -1.6, 0.5] for p in test_points_outside], dtype=np.float32)
        cells = grid.build_foveated(pts_arr)

        for c in cells:
            assert c.zone not in (1, 2, 3, 4), (
                f"Cell {c.cell_id} at ({c.x}, {c.y}) was incorrectly assigned perception zone {c.zone}!"
            )

    @pytest.mark.parametrize(
        "dist, expected_zone, expected_res",
        [
            (0.00, 1, 0.05),     # Origin
            (5.00, 1, 0.05),     # Zone 1 interior
            (9.99, 1, 0.05),     # Zone 1 just inside boundary
            (10.00, 2, 0.10),    # Zone 2 boundary
            (10.01, 2, 0.10),    # Zone 2 just outside boundary
            (24.99, 2, 0.10),    # Zone 2 just inside boundary
            (25.00, 3, 0.25),    # Zone 3 boundary
            (25.01, 3, 0.25),    # Zone 3 just outside boundary
            (49.99, 3, 0.25),    # Zone 3 just inside boundary
            (50.00, 4, 0.50),    # Zone 4 boundary
            (50.01, 4, 0.50),    # Zone 4 just outside boundary
            (99.99, 4, 0.50),    # Zone 4 just inside boundary
            (100.00, 4, 0.50),   # Exactly at 100m envelope
            (100.01, 5, 0.50),   # Just beyond 100m envelope -> Zone 5 (outside)
            (105.00, 5, 0.50),   # Beyond envelope -> Zone 5 (outside)
        ],
    )
    def test_zone_boundary_conditions(self, dist, expected_zone, expected_res):
        """Audit Item 7 & 14: Comprehensive Zone Boundary Condition Tests."""
        zone = get_zone_for_distance(dist)
        res = get_mandated_resolution(zone)
        assert zone == expected_zone, f"At {dist}m expected zone {expected_zone}, got {zone}"
        assert abs(res - expected_res) < 1e-6, f"At {dist}m expected res {expected_res}, got {res}"

    def test_no_zone_overlap(self):
        """Audit Item 14: Verification that no cells overlap across zone transitions."""
        grid = FoveatedGrid(config=FOVEATED, bounds=MapBounds)
        # Create a dense radial ray crossing all zone boundaries
        radii = [5.0, 9.98, 10.02, 18.0, 24.98, 25.02, 38.0, 49.98, 50.02, 85.0, 99.98]
        pts = np.array([[r, 0.0, -1.6, 0.5] for r in radii], dtype=np.float32)
        cells = grid.build_foveated(pts)

        n = len(cells)
        for i in range(n):
            for j in range(i + 1, n):
                c1, c2 = cells[i], cells[j]
                overlap_x = min(c1.x_max, c2.x_max) - max(c1.x_min, c2.x_min)
                overlap_y = min(c1.y_max, c2.y_max) - max(c1.y_min, c2.y_min)
                assert not (overlap_x > 1e-5 and overlap_y > 1e-5), (
                    f"Cells {c1.cell_id} and {c2.cell_id} overlap! "
                    f"overlap_x={overlap_x}, overlap_y={overlap_y}"
                )

    def test_no_zone_straddling(self):
        """Audit Item 14: Verification that no cell straddles a zone boundary."""
        grid = FoveatedGrid(config=FOVEATED, bounds=MapBounds)
        boundary_radii = [9.99, 10.0, 10.01, 24.99, 25.0, 25.01, 49.99, 50.0, 50.01, 99.99, 100.0]
        pts = np.array([[r, 0.0, -1.6, 0.5] for r in boundary_radii], dtype=np.float32)
        cells = grid.build_foveated(pts)

        for c in cells:
            r_min, r_max = cell_footprint_r_min_r_max(c.x_min, c.x_max, c.y_min, c.y_max)
            straddles = check_zone_straddling(r_min, r_max)
            assert not straddles, (
                f"Cell {c.cell_id} straddles radial zone boundaries: r_min={r_min:.4f}, r_max={r_max:.4f}"
            )

