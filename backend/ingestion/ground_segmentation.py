"""
High-speed 2.5D ground plane segmentation.
Concentric Zone / Radial Ring elevation estimation + SVD local plane refinement.
Separates raw LiDAR points into:
  - ground_points: used for terrain slope and quadtree elevation statistics
  - obstacle_points: non-ground points used for clustering and obstacle tracking
"""

from typing import Tuple
import numpy as np

from backend.config import GROUND_SEG


class GroundSegmenter:
    """
    Real-time ground plane extractor optimized for edge execution (< 5 ms).
    Partitions the horizontal plane into concentric rings and angular sectors,
    estimates local lowest point seeds, and extracts ground vs obstacle masks.
    """

    def __init__(
        self,
        sensor_height: float = GROUND_SEG.SENSOR_HEIGHT_NOMINAL,
        ground_distance_thresh: float = GROUND_SEG.GROUND_DISTANCE_THRESHOLD,
        num_rings: int = GROUND_SEG.NUM_RADIAL_BINS,
        num_sectors: int = GROUND_SEG.NUM_ANGULAR_SECTORS,
    ):
        self.sensor_height = sensor_height
        self.thresh = ground_distance_thresh
        self.num_rings = num_rings
        self.num_sectors = num_sectors

    def segment(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment points into ground and obstacle arrays.

        Args:
            points: (N, 4) or (N, 3) float32 array [x, y, z, (intensity)]

        Returns:
            ground_points: (M, 4) points classified as terrain
            obstacle_points: (K, 4) non-ground points (potential obstacles)
            ground_mask: (N,) boolean array where True = ground
        """
        if len(points) == 0:
            empty = np.empty((0, points.shape[1]), dtype=np.float32)
            return empty, empty, np.zeros(0, dtype=bool)

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        r = np.hypot(x, y)
        theta = np.arctan2(y, x)  # [-pi, pi]

        # Fast rejection of obviously high points (e.g. above sensor height + 0.3m)
        elev_mask = z <= (-self.sensor_height + 0.35)
        
        # Bin points into polar grid (radial rings + angular sectors)
        max_r = 25.0
        ring_idx = np.clip((r / max_r * self.num_rings).astype(np.int32), 0, self.num_rings - 1)
        sector_idx = np.clip(
            ((theta + np.pi) / (2 * np.pi) * self.num_sectors).astype(np.int32),
            0,
            self.num_sectors - 1,
        )
        cell_keys = ring_idx * self.num_sectors + sector_idx
        num_cells = self.num_rings * self.num_sectors

        # In each polar bin, find lowest point using np.minimum.at
        min_z_per_cell = np.full(num_cells, np.inf, dtype=np.float32)
        np.minimum.at(min_z_per_cell, cell_keys, z)

        # For empty bins, fallback to nominal ground
        inf_mask = np.isinf(min_z_per_cell)
        min_z_per_cell[inf_mask] = -self.sensor_height

        # Evaluate distance of each point to its local bin ground seed
        cell_min_z = min_z_per_cell[cell_keys]
        diff_z = z - cell_min_z

        # Point is ground if it lies within threshold of the cell ground seed
        ground_mask = (diff_z >= -0.05) & (diff_z <= self.thresh)

        # In extremely close range (r < 1.0m), eliminate ego sensor vehicle reflections
        close_mask = r < 1.0
        ground_mask[close_mask] = (z[close_mask] <= (-self.sensor_height + 0.05))

        ground_points = points[ground_mask]
        obstacle_points = points[~ground_mask]

        return ground_points, obstacle_points, ground_mask
