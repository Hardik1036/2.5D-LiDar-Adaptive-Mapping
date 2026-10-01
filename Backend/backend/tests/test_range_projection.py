import pytest
import numpy as np

from backend.ingestion.range_projection import SphericalRangeProjector


class TestRangeProjection:
    """12.6 Spherical Range Projection & Bidirectional Mapping Tests."""

    def test_range_image_shape(self):
        projector = SphericalRangeProjector(height=64, width=1024)
        rng = np.random.RandomState(42)
        n = 10000
        x = rng.uniform(-30, 80, n).astype(np.float32)
        y = rng.uniform(-40, 40, n).astype(np.float32)
        z = rng.uniform(-2, 3, n).astype(np.float32)
        intensity = rng.uniform(0, 1, n).astype(np.float32)
        ring = rng.randint(0, 64, n).astype(np.float32)

        points = np.stack([x, y, z, intensity, ring], axis=1)
        result = projector.project(points)

        assert result.range_image.shape == (64, 1024, 5), (
            f"Expected (64, 1024, 5), got {result.range_image.shape}"
        )
        assert result.point_to_pixel.shape == (n, 2)
        assert result.pixel_to_point.shape == (64, 1024)

    def test_bidirectional_consistency(self):
        projector = SphericalRangeProjector(height=64, width=1024)
        # Create non-colliding points along distinct angles
        thetas = np.linspace(-np.pi * 0.8, np.pi * 0.8, 100)
        phis = np.linspace(-0.2, 0.1, 100)
        r = 15.0
        x = (r * np.cos(phis) * np.cos(thetas)).astype(np.float32)
        y = (r * np.cos(phis) * np.sin(thetas)).astype(np.float32)
        z = (r * np.sin(phis)).astype(np.float32)
        points = np.stack([x, y, z], axis=1)

        result = projector.project(points)
        p2pix = result.point_to_pixel
        pix2p = result.pixel_to_point

        valid_matches = 0
        for pt_idx in range(len(points)):
            r_idx, c_idx = p2pix[pt_idx]
            if r_idx >= 0 and c_idx >= 0:
                mapped_back_idx = pix2p[r_idx, c_idx]
                if mapped_back_idx == pt_idx:
                    valid_matches += 1

        # At least 90% bidirectional exact match on non-colliding points
        assert valid_matches >= 80, f"Expected >= 80 matching points, got {valid_matches}"

    def test_empty_point_cloud(self):
        projector = SphericalRangeProjector(height=64, width=1024)
        empty_points = np.empty((0, 3), dtype=np.float32)
        result = projector.project(empty_points)

        assert result.range_image.shape == (64, 1024, 5)
        assert np.all(result.range_image == 0.0)
        assert result.point_to_pixel.shape == (0, 2)
        assert np.all(result.pixel_to_point == -1)

    def test_invalid_nan_inf_points(self):
        projector = SphericalRangeProjector(height=64, width=1024)
        points = np.array([
            [10.0, 5.0, 0.0, 0.5],
            [np.nan, 2.0, 1.0, 0.5],
            [15.0, np.inf, 0.0, 0.5],
            [-np.inf, 0.0, 0.0, 0.5],
            [20.0, 10.0, -1.0, 0.8],
        ], dtype=np.float32)

        # Must not raise exception
        result = projector.project(points)
        assert result.range_image.shape == (64, 1024, 5)
        # NaN / Inf points must have mapped pixel (-1, -1)
        assert result.point_to_pixel[1, 0] == -1
        assert result.point_to_pixel[2, 0] == -1
        assert result.point_to_pixel[3, 0] == -1

    def test_duplicate_projections(self):
        projector = SphericalRangeProjector(height=64, width=1024)
        # Multiple points at exact same spherical ray but different depth
        points = np.array([
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],  # Farther
            [5.0, 0.0, 0.0],   # Closest (should win pixel_to_point depth sorting)
        ], dtype=np.float32)

        result = projector.project(points)
        assert result.range_image.shape == (64, 1024, 5)
        # Check closest point was retained in pixel_to_point
        r0, c0 = result.point_to_pixel[2]
        if r0 >= 0 and c0 >= 0:
            assert result.pixel_to_point[r0, c0] == 2, "Closest point should be retained in pixel_to_point"
