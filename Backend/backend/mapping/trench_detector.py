"""
PS 26053 Local-Baseline Negative Obstacle & Trench Detector.

Uses a local 3x3 neighbourhood median baseline:
    Z_local_baseline(x, y) = median(valid neighbouring Z_mean values)
    drop = Z_local_baseline - Z_cell

A cell is a negative obstacle only when:
    drop > 0.15 m
relative to the local baseline.

Prevents false positives on:
- Uniform slopes (e.g. 15 deg road grade has drop ~ 0.0m <= 0.15m)
- Ramps and smooth highway transitions
- Legitimate global elevation shifts
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from backend.mapping.quadtree import QuadtreeNode


def detect_local_baseline_dropoffs(
    cells: List[Any],
    drop_threshold_m: float = 0.15,
    res: float = 0.50,
) -> Dict[int, float]:
    """
    Evaluates each cell against its 3x3 neighbourhood median baseline:
        Z_local_baseline = median(valid neighbouring Z_mean)
        drop = Z_local_baseline - Z_cell
    Returns a mapping from cell index to drop value (in meters).
    """
    if not cells:
        return {}

    grid: Dict[Tuple[int, int], List[Tuple[int, Any]]] = {}
    for idx, c in enumerate(cells):
        cx = c.get("x", 0.0) if isinstance(c, dict) else getattr(c, "x", 0.0)
        cy = c.get("y", 0.0) if isinstance(c, dict) else getattr(c, "y", 0.0)
        gx = int(np.floor(cx / res))
        gy = int(np.floor(cy / res))
        grid.setdefault((gx, gy), []).append((idx, c))

    drops: Dict[int, float] = {}

    for (gx, gy), indexed_cells in grid.items():
        neighbour_z: List[float] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nbrs = grid.get((gx + dx, gy + dy))
                if nbrs:
                    for _, nc in nbrs:
                        z_val = nc.get("z_mean", nc.get("z", None)) if isinstance(nc, dict) else getattr(nc, "z_mean", getattr(nc, "z", None))
                        if z_val is not None and np.isfinite(z_val):
                            neighbour_z.append(float(z_val))

        if len(neighbour_z) < 3:
            continue

        baseline_z = float(np.median(neighbour_z))

        for idx, c in indexed_cells:
            z_val = c.get("z_mean", c.get("z", baseline_z)) if isinstance(c, dict) else getattr(c, "z_mean", getattr(c, "z", baseline_z))
            drop = baseline_z - float(z_val)
            drops[idx] = float(drop)

    return drops


class TrenchDetector:
    """
    Dual-mode Trench and Negative Obstacle Detector:
    1. Line-of-sight ray shadow analysis for raw point clouds (Feature F5.1).
    2. Local 3x3 neighbourhood baseline evaluation on 2.5D cell elevations (PS 26053).
    """

    def __init__(
        self,
        min_elevation_drop: float = 0.15,   # PS 26053 standard: 0.15 m drop threshold
        drop_threshold_m: Optional[float] = None,
        min_radial_gap: float = 0.40,       # 40 cm missing return gap along ray
        num_angular_sectors: int = 64,      # Angular polar discretization
        hazard_radius: float = 0.45,
        forced_cost: int = 255,
        grid_resolution: float = 0.50,      # Neighbourhood binning resolution
    ):
        self.min_drop = drop_threshold_m if drop_threshold_m is not None else min_elevation_drop
        self.min_gap = min_radial_gap
        self.num_sectors = num_angular_sectors
        self.hazard_radius = hazard_radius
        self.forced_cost = forced_cost
        self.grid_res = grid_resolution

    def detect_negative_obstacles(
        self,
        cells: List[Any],
        drop_threshold: Optional[float] = None,
    ) -> List[int]:
        """Returns list of cell indices identified as local negative obstacles."""
        thresh = drop_threshold if drop_threshold is not None else self.min_drop
        drops = detect_local_baseline_dropoffs(cells, drop_threshold_m=thresh, res=self.grid_res)
        return [idx for idx, drop in drops.items() if drop > thresh]

    def find_dropoffs(
        self,
        points: np.ndarray,
        sensor_origin: np.ndarray = np.array([0.0, 0.0, 1.5]),
    ) -> List[Dict[str, Any]]:
        """
        Scans ground returns along radial rays to find line-of-sight drop-offs.
        Preserves compatibility with legacy ray dropout pipeline.
        """
        if points is None or len(points) < 50:
            return []

        ox, oy, oz = float(sensor_origin[0]), float(sensor_origin[1]), float(sensor_origin[2])
        px = points[:, 0] - ox
        py = points[:, 1] - oy
        pz = points[:, 2]

        radii = np.hypot(px, py)
        angles = np.arctan2(py, px)

        bin_width = (2.0 * np.pi) / self.num_sectors
        sector_bins = np.floor((angles + np.pi) / bin_width).astype(np.int32)
        sector_bins = np.clip(sector_bins, 0, self.num_sectors - 1)

        order = np.lexsort((radii, sector_bins))
        s_sorted = sector_bins[order]
        r_sorted = radii[order]
        z_sorted = pz[order]
        x_sorted = points[order, 0]
        y_sorted = points[order, 1]

        same_sector = s_sorted[1:] == s_sorted[:-1]
        delta_r = r_sorted[1:] - r_sorted[:-1]
        delta_z = z_sorted[:-1] - z_sorted[1:]

        trench_mask = same_sector & (delta_r >= self.min_gap) & (delta_z >= self.min_drop)
        trench_indices = np.nonzero(trench_mask)[0]

        dropoffs = []
        for t_idx in trench_indices:
            dropoffs.append({
                "type": "negative_obstacle",
                "x": float(x_sorted[t_idx]),
                "y": float(y_sorted[t_idx]),
                "z": float(z_sorted[t_idx]),
                "drop_depth": float(delta_z[t_idx]),
                "radius": self.hazard_radius,
                "cost": self.forced_cost,
            })

        return dropoffs

    def detect_local_baseline_dropoffs(
        self,
        leaves: List[Any],
        drop_threshold: float = 0.15,
    ) -> List[Dict[str, Any]]:
        """
        PS 26053 Local 3x3 Terrain Baseline Negative Obstacle Detection.
        For every leaf cell:
            1. Query 3x3 local spatial neighborhood of valid neighboring cells
            2. Compute Z_local_baseline = median(valid neighboring Z_mean)
            3. drop = Z_local_baseline - Z_cell
            4. If drop > drop_threshold (0.15m), mark as negative obstacle

        Guarantees that a uniform 15 deg slope is NOT flagged as a negative obstacle,
        while a true local 0.20m ditch/trench depression IS detected.
        """
        if not leaves:
            return []

        # 1. Map cells into 2D spatial hash grid
        res = self.grid_res
        grid: Dict[Tuple[int, int], List[Any]] = {}
        for leaf in leaves:
            gx = int(np.floor(leaf.x / res))
            gy = int(np.floor(leaf.y / res))
            grid.setdefault((gx, gy), []).append(leaf)

        detected_dropoffs: List[Dict[str, Any]] = []

        # 2. Evaluate each leaf against its 3x3 neighbourhood
        for (gx, gy), cell_list in grid.items():
            # Collect valid neighbouring Z_mean values across 3x3 window
            neighbour_z: List[float] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nbr_cells = grid.get((gx + dx, gy + dy))
                    if nbr_cells:
                        for nc in nbr_cells:
                            nc_z = getattr(nc, "z_mean", getattr(nc, "z", None))
                            if nc_z is not None and np.isfinite(nc_z):
                                neighbour_z.append(float(nc_z))

            if len(neighbour_z) < 3:
                continue

            z_local_baseline = float(np.median(neighbour_z))

            for leaf in cell_list:
                z_cell = float(getattr(leaf, "z_mean", getattr(leaf, "z", z_local_baseline)))
                drop = z_local_baseline - z_cell

                if drop > drop_threshold:
                    detected_dropoffs.append({
                        "type": "negative_obstacle_local",
                        "x": float(leaf.x),
                        "y": float(leaf.y),
                        "z": float(z_cell),
                        "baseline_z": float(z_local_baseline),
                        "drop_depth": float(drop),
                        "cost": self.forced_cost,
                    })

        return detected_dropoffs

    def apply_dropoffs_to_leaves(
        self,
        leaves: List[QuadtreeNode],
        dropoffs: List[Dict[str, Any]],
    ) -> None:
        """
        Marks Quadtree leaves overlapping negative obstacles as lethal (cost = 255).
        """
        if not leaves or not dropoffs:
            return

        for d in dropoffs:
            dx_c, dy_c = d["x"], d["y"]
            r_sq = (d.get("radius", self.hazard_radius)) ** 2
            cost = d.get("cost", self.forced_cost)

            for leaf in leaves:
                dist_sq = (leaf.x - dx_c) ** 2 + (leaf.y - dy_c) ** 2
                if dist_sq <= r_sq:
                    leaf.cost = cost
                    leaf.is_hazard = True
                    leaf.is_obstacle = True
                    if hasattr(leaf, "semantic_cost"):
                        leaf.semantic_cost = cost
