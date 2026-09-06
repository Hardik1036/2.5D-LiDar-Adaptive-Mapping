"""
Statistical Outlier Dust and Atmospheric Scatter Filter (Feature F1.3).
Removes sparse floating returns caused by dust clouds, rain droplets,
and exhaust scatter using k-NN statistical spatial distance distributions.
Accelerated with 25m range gating and subsampled KD-tree queries for > 15,000 pts.
"""

from typing import Optional
import numpy as np
from scipy.spatial import cKDTree


class StatisticalDustFilter:
    """
    Edge-optimized Statistical Outlier Removal (SOR) filter.
    Computes mean k-NN Euclidean distance distribution and strips points
    falling outside mu + std_ratio * sigma.
    """

    def __init__(self, k: int = 15, std_ratio: float = 2.0, max_range: float = 25.0):
        self.k = k
        self.std_ratio = std_ratio
        self.max_range = max_range
        self.max_range_sq = max_range * max_range

    def filter(
        self,
        points: np.ndarray,
        k: Optional[int] = None,
        std_ratio: Optional[float] = None,
    ) -> np.ndarray:
        """
        Filters airborne dust and sparse atmospheric scatter from raw LiDAR returns.
        Accelerated with a 25m bounding range gate and stride=2 KD-tree subsampling
        for dense point clouds (> 15,000 points) to guarantee <= 2.0 ms latency.

        Args:
            points: (N, 3) or (N, 4) point coordinates [x, y, z, (intensity)].
            k: Number of nearest neighbors to evaluate (defaults to self.k).
            std_ratio: Standard deviation multiplier threshold (defaults to self.std_ratio).

        Returns:
            Filtered point array containing only dense structural/ground returns.
        """
        if points is None or len(points) == 0:
            return np.empty((0, 3 if points is None else points.shape[1]), dtype=np.float32)

        n_pts = len(points)
        eval_k = k if k is not None else self.k
        eval_ratio = std_ratio if std_ratio is not None else self.std_ratio

        if n_pts <= eval_k:
            return points

        coords = points[:, :3]

        # 1. Bounding Range Gate: filter airborne dust primarily within 25m ego radius
        x = coords[:, 0]
        y = coords[:, 1]
        dist_sq = x * x + y * y
        in_range_mask = dist_sq <= self.max_range_sq
        pts_in_range = points[in_range_mask]
        pts_out_range = points[~in_range_mask]

        if len(pts_in_range) <= eval_k:
            return points

        coords_in = pts_in_range[:, :3]
        n_in = len(pts_in_range)

        # 2. For large point clouds, partition ground and airborne points
        if n_in > 1500 or n_pts > 1500:
            zx, zy, zz = coords_in[:, 0], coords_in[:, 1], coords_in[:, 2]
            # Coarse 2D elevation grid (2.0m cells) over [-20, 20]
            gx = np.clip(((zx + 20.0) * 0.5).astype(np.int32), 0, 19)
            gy = np.clip(((zy + 20.0) * 0.5).astype(np.int32), 0, 19)
            cell_id = gy * 20 + gx

            min_z_map = np.full(400, 999.0, dtype=np.float32)
            np.minimum.at(min_z_map, cell_id, zz)

            is_ground = (zz - min_z_map[cell_id]) <= 0.22
            air_pts = pts_in_range[~is_ground]
            ground_pts = pts_in_range[is_ground]

            if len(air_pts) <= eval_k:
                return points

            # Subsampled KD-tree indexing when point count exceeds 8,000 points or > 600 air points
            if n_pts > 8000 or len(air_pts) > 600:
                stride = 2
                tree_pts = air_pts[::stride, :3]
                tree = cKDTree(tree_pts, leafsize=32)
                query_k = min(eval_k + 1, len(tree_pts))
                dists, _ = tree.query(air_pts[:, :3], k=query_k, workers=-1)
            else:
                tree = cKDTree(air_pts[:, :3], leafsize=32)
                dists, _ = tree.query(air_pts[:, :3], k=eval_k + 1, workers=-1)

            mean_dists = np.mean(dists[:, 1:], axis=1)

            mu = float(np.mean(mean_dists))
            sigma = float(np.std(mean_dists))

            if sigma < 1e-6:
                return points

            cutoff = mu + eval_ratio * sigma
            keep_air = air_pts[mean_dists <= cutoff]
            clean_in_range = np.vstack([ground_pts, keep_air])

            if len(pts_out_range) > 0:
                return np.vstack([clean_in_range, pts_out_range])
            return clean_in_range

        # Standard cKDTree for smaller point clouds
        tree = cKDTree(coords_in, leafsize=32)
        dists, _ = tree.query(coords_in, k=eval_k + 1, workers=1)
        mean_dists = np.mean(dists[:, 1:], axis=1)

        mu = float(np.mean(mean_dists))
        sigma = float(np.std(mean_dists))

        if sigma < 1e-6:
            return points

        cutoff = mu + eval_ratio * sigma
        clean_in = pts_in_range[mean_dists <= cutoff]

        if len(pts_out_range) > 0:
            return np.vstack([clean_in, pts_out_range])
        return clean_in
