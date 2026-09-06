"""
Negative Obstacle & Trench Drop-off Detector (Feature F5.1).
Exposes blind negative terrain hazards (ditches, drop-offs, trenches, erosion gullies)
using geometric ray shadow and line-of-sight dropout analysis.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from backend.mapping.quadtree import QuadtreeNode


class TrenchDetector:
    """
    Detects negative obstacles that do not produce positive elevation returns.
    Identifies ray dropout shadows where ground drops abruptly (> 0.25m) along line of sight.
    """

    def __init__(
        self,
        min_elevation_drop: float = 0.25,   # 25 cm drop threshold
        min_radial_gap: float = 0.40,       # 40 cm missing return gap along ray
        num_angular_sectors: int = 64,      # Angular polar discretization
        hazard_radius: float = 0.45,
        forced_cost: int = 255,
    ):
        self.min_drop = min_elevation_drop
        self.min_gap = min_radial_gap
        self.num_sectors = num_angular_sectors
        self.hazard_radius = hazard_radius
        self.forced_cost = forced_cost

    def find_dropoffs(
        self,
        points: np.ndarray,
        sensor_origin: np.ndarray = np.array([0.0, 0.0, 1.5]),
    ) -> List[Dict[str, Any]]:
        """
        Scans ground returns along radial rays to find line-of-sight drop-offs.

        Args:
            points: (N, 3) or (N, 4) point cloud.
            sensor_origin: (3,) LiDAR optical center coordinates.

        Returns:
            List of detected drop-off hazard records with coordinates and severity.
        """
        if points is None or len(points) < 50:
            return []

        ox, oy, oz = float(sensor_origin[0]), float(sensor_origin[1]), float(sensor_origin[2])
        px = points[:, 0] - ox
        py = points[:, 1] - oy
        pz = points[:, 2]

        radii = np.hypot(px, py)
        angles = np.arctan2(py, px)  # [-pi, pi]

        # Discretize angles into sector bins
        bin_width = (2.0 * np.pi) / self.num_sectors
        sector_bins = np.floor((angles + np.pi) / bin_width).astype(np.int32)
        sector_bins = np.clip(sector_bins, 0, self.num_sectors - 1)

        dropoffs = []

        # Analyze each angular sector
        for s in range(self.num_sectors):
            mask = sector_bins == s
            n_in_sector = np.count_nonzero(mask)
            if n_in_sector < 5:
                continue

            sec_r = radii[mask]
            sec_z = pz[mask]
            sec_x = points[mask, 0]
            sec_y = points[mask, 1]

            # Sort ascending by radial distance
            sort_idx = np.argsort(sec_r)
            r_sorted = sec_r[sort_idx]
            z_sorted = sec_z[sort_idx]
            x_sorted = sec_x[sort_idx]
            y_sorted = sec_y[sort_idx]

            # Vectorized radial diffs and elevation diffs
            delta_r = r_sorted[1:] - r_sorted[:-1]
            delta_z = z_sorted[:-1] - z_sorted[1:]  # positive when next point drops lower

            # Trench condition: radial distance jump AND lower elevation
            trench_indices = np.where((delta_r >= self.min_gap) & (delta_z >= self.min_drop))[0]

            for t_idx in trench_indices:
                edge_x = float(x_sorted[t_idx])
                edge_y = float(y_sorted[t_idx])
                edge_z = float(z_sorted[t_idx])
                drop_val = float(delta_z[t_idx])

                dropoffs.append({
                    "type": "negative_obstacle",
                    "x": edge_x,
                    "y": edge_y,
                    "z": edge_z,
                    "drop_depth": drop_val,
                    "radius": self.hazard_radius,
                    "cost": self.forced_cost,
                })

        return dropoffs

    def apply_dropoffs_to_leaves(
        self,
        leaves: List[QuadtreeNode],
        dropoffs: List[Dict[str, Any]],
    ):
        """
        Marks Quadtree leaves overlapping negative obstacles as lethal (cost = 255).
        """
        if not leaves or not dropoffs:
            return

        for drop in dropoffs:
            dx_c = drop["x"]
            dy_c = drop["y"]
            r2 = drop["radius"] * drop["radius"]
            hz_cost = drop.get("cost", self.forced_cost)

            for leaf in leaves:
                dist2 = (leaf.x - dx_c) ** 2 + (leaf.y - dy_c) ** 2
                if dist2 <= r2:
                    leaf.cost = max(leaf.cost, hz_cost)
