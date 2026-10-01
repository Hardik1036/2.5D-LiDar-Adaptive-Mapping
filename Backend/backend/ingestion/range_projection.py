"""
Spherical Range Projection & Bidirectional Point-Pixel Mapping.
Converts 3D LiDAR point clouds (N, 3/4/5) to spherical range image tensors (64, 1024, 5)
and maps 2D deep learning semantic predictions back to 3D points with strict correspondence.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class ProjectionConfig:
    """Spherical projection geometry parameters."""
    H: int = 64
    W: int = 1024
    fov_up_deg: float = 17.02
    fov_down_deg: float = -16.44

    @property
    def fov_up_rad(self) -> float:
        return float(np.radians(self.fov_up_deg))

    @property
    def fov_down_rad(self) -> float:
        return float(np.radians(self.fov_down_deg))

    @property
    def fov_total_rad(self) -> float:
        return self.fov_up_rad - self.fov_down_rad


@dataclass
class RangeProjectionResult:
    """PS 26053 Range Projection Result supporting both tuple unpacking and attribute access."""
    range_image: np.ndarray
    point_to_pixel: np.ndarray
    pixel_to_point: np.ndarray
    valid_mask: np.ndarray

    def __iter__(self):
        return iter((self.range_image, self.point_to_pixel, self.pixel_to_point, self.valid_mask))

    def __getitem__(self, idx):
        return (self.range_image, self.point_to_pixel, self.pixel_to_point, self.valid_mask)[idx]


class SphericalRangeProjector:
    """
    Spherical range projection utility for 64x1024 LiDAR sweeps.
    Produces (64, 1024, 5) [Range, X, Y, Z, Intensity] range images.
    Maintains bidirectional point_to_pixel and pixel_to_point indexing.
    """

    def __init__(
        self,
        config: Optional[ProjectionConfig] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ):
        self.config = config or ProjectionConfig()
        self.H = height if height is not None else self.config.H
        self.W = width if width is not None else self.config.W
        self.fov_up = self.config.fov_up_rad
        self.fov_down = self.config.fov_down_rad
        self.fov_total = self.config.fov_total_rad

    def project(
        self, points: np.ndarray
    ) -> RangeProjectionResult:
        """
        Projects an (N, 3), (N, 4), or (N, 5) point cloud to a 64x1024x5 spherical range image.

        Returns:
            RangeProjectionResult (tuple-unpackable and with attributes):
                range_image: (H, W, 5) float32 array [r, x, y, z, intensity]
                point_to_pixel: (N, 2) int32 array of (v, u) image coordinates for each 3D point
                pixel_to_point: (H, W) int32 array of closest 3D point indices (-1 for empty pixels)
                valid_mask: (N,) bool array indicating valid in-FOV points
        """
        N = len(points) if points is not None else 0

        # Handle empty sweeps
        if N == 0:
            range_image = np.zeros((self.H, self.W, 5), dtype=np.float32)
            point_to_pixel = np.full((0, 2), -1, dtype=np.int32)
            pixel_to_point = np.full((self.H, self.W), -1, dtype=np.int32)
            valid_mask = np.zeros(0, dtype=bool)
            return RangeProjectionResult(range_image, point_to_pixel, pixel_to_point, valid_mask)

        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 3 if pts.shape[0] % 3 == 0 else 4)

        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        intensity = pts[:, 3] if pts.shape[1] >= 4 else np.full(N, 0.5, dtype=np.float32)

        # Range r
        r = np.sqrt(x * x + y * y + z * z)
        r_safe = np.maximum(r, 1e-6)

        # Yaw and Pitch spherical projection angles
        yaw = -np.arctan2(y, x)
        pitch = np.arcsin(np.clip(z / r_safe, -1.0, 1.0))

        # Horizontal coordinate u in [0, W - 1]
        u = 0.5 * ((yaw / np.pi) + 1.0) * self.W

        # Vertical coordinate v in [0, H - 1]
        v = (1.0 - (pitch + abs(self.fov_down)) / self.fov_total) * self.H

        # Boundary validation & clamping with NaN/Inf safety
        finite_mask = np.isfinite(u) & np.isfinite(v) & np.isfinite(r)
        u_safe = np.where(finite_mask, u, 0.0)
        v_safe = np.where(finite_mask, v, 0.0)

        u_int = np.floor(u_safe).astype(np.int32)
        v_int = np.floor(v_safe).astype(np.int32)

        valid_mask = (
            finite_mask &
            (v_int >= 0) & (v_int < self.H) &
            (u_int >= 0) & (u_int < self.W) &
            (r > 0.1) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        )

        u_clamped = np.clip(u_int, 0, self.W - 1)
        v_clamped = np.clip(v_int, 0, self.H - 1)

        point_to_pixel = np.column_stack((v_clamped, u_clamped)).astype(np.int32)
        # Mark invalid points with -1
        point_to_pixel[~valid_mask] = -1

        # Initialize output buffers
        range_image = np.zeros((self.H, self.W, 5), dtype=np.float32)
        pixel_to_point = np.full((self.H, self.W), -1, dtype=np.int32)

        valid_indices = np.nonzero(valid_mask)[0]
        if len(valid_indices) > 0:
            # Sort valid points by decreasing range so closer points overwrite further points (collision handling)
            depth_order = valid_indices[np.argsort(-r[valid_indices])]
            vo = v_clamped[depth_order]
            uo = u_clamped[depth_order]

            range_image[vo, uo, 0] = r[depth_order]
            range_image[vo, uo, 1] = x[depth_order]
            range_image[vo, uo, 2] = y[depth_order]
            range_image[vo, uo, 3] = z[depth_order]
            range_image[vo, uo, 4] = intensity[depth_order]

            pixel_to_point[vo, uo] = depth_order

        return RangeProjectionResult(range_image, point_to_pixel, pixel_to_point, valid_mask)

    def to_tensor_format(self, range_image: np.ndarray) -> np.ndarray:
        """
        Converts (H, W, 5) range image to (1, 5, H, W) NCHW float32 tensor
        as expected by ONNX / PyTorch segmentation models.
        """
        # Transpose (H, W, 5) -> (5, H, W) -> (1, 5, H, W)
        chw = np.transpose(range_image, (2, 0, 1))
        return np.expand_dims(chw, axis=0).astype(np.float32)
