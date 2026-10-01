import pytest
import numpy as np

from backend.mapping.trench_detector import TrenchDetector, detect_local_baseline_dropoffs


class TestLocalNegativeObstacles:
    """12.7 Local Terrain Baseline Negative Obstacle Detection Tests."""

    def test_uniform_15_deg_slope_is_not_negative_obstacle(self):
        """A uniform 15 degree slope must NOT be detected as a negative obstacle."""
        detector = TrenchDetector(drop_threshold_m=0.15)
        angle_rad = np.deg2rad(15.0)

        # Build a 7x7 grid with a continuous 15° slope along X
        # slope: z = -x * tan(15°)
        cells = []
        res = 0.5
        for ix in range(7):
            for iy in range(7):
                x = (ix - 3) * res
                y = (iy - 3) * res
                z = float(-x * np.tan(angle_rad))
                cells.append({
                    "x": x,
                    "y": y,
                    "size": res,
                    "z_mean": z,
                    "z_min": z - 0.02,
                    "z_max": z + 0.02,
                })

        drops = detect_local_baseline_dropoffs(cells, drop_threshold_m=0.15)
        num_negative_obstacles = sum(1 for d in drops.values() if d > 0.15)
        assert num_negative_obstacles == 0, (
            f"Uniform 15° slope produced {num_negative_obstacles} false negative obstacles!"
        )

    def test_local_20cm_depression_is_negative_obstacle(self):
        """A local 0.20m depression relative to 3x3 local median must be detected as a negative obstacle."""
        detector = TrenchDetector(drop_threshold_m=0.15)

        # Build a flat 5x5 plane at z = -1.60m, with center cell at z = -1.85m (0.25m drop)
        cells = []
        res = 0.5
        for ix in range(5):
            for iy in range(5):
                x = (ix - 2) * res
                y = (iy - 2) * res
                is_center = (ix == 2 and iy == 2)
                z = -1.85 if is_center else -1.60
                cells.append({
                    "x": x,
                    "y": y,
                    "size": res,
                    "z_mean": z,
                    "z_min": z - 0.02,
                    "z_max": z + 0.02,
                })

        drops = detect_local_baseline_dropoffs(cells, drop_threshold_m=0.15)
        # Center cell is index 2*5 + 2 = 12
        center_drop = drops.get(12, 0.0)
        assert center_drop >= 0.20, f"Expected center drop >= 0.20m, got {center_drop:.3f}m"

        neg_indices = detector.detect_negative_obstacles(cells)
        assert 12 in neg_indices, "Center depression (0.25m drop) was not detected as negative obstacle"
        # None of the outer cells should be negative obstacles
        assert len(neg_indices) == 1, f"Expected exactly 1 negative obstacle, got {len(neg_indices)}"
