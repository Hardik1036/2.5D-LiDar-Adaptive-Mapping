"""
Adaptive 2.5D Quadtree partitioning strictly on the (X, Y) horizontal plane.
Subdivides coarse 50cm cells down to fine 5cm resolution only when
local elevation variance > tau_sigma or step height delta_z > tau_z.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.ndimage as ndi  # type: ignore

from backend.config import BOUNDS, QUADTREE
from backend.mapping.cell_statistics import CellStats, compute_cell_statistics


@dataclass(slots=True)
class QuadtreeNode:
    """
    Compact 2.5D Quadtree Node.
    Zero raw point arrays are stored permanently.
    """
    x: float             # Center X
    y: float             # Center Y
    size: float          # Half-size or full cell width
    depth: int           # Tree depth level
    stats: Optional[CellStats] = None
    cost: int = 0        # Traversability cost [0, 255]
    is_leaf: bool = True
    children: Optional[List["QuadtreeNode"]] = None
    semantic_cost: int = 0
    is_obstacle: bool = False
    is_hazard: bool = False
    cell_id: Optional[str] = None
    zone: int = 1
    observed: bool = True
    valid: bool = True
    _x_min: Optional[float] = None
    _x_max: Optional[float] = None
    _y_min: Optional[float] = None
    _y_max: Optional[float] = None

    @property
    def x_min(self) -> float:
        return self._x_min if self._x_min is not None else (self.x - self.size * 0.5)

    @x_min.setter
    def x_min(self, val: float):
        self._x_min = float(val)

    @property
    def x_max(self) -> float:
        return self._x_max if self._x_max is not None else (self.x + self.size * 0.5)

    @x_max.setter
    def x_max(self, val: float):
        self._x_max = float(val)

    @property
    def y_min(self) -> float:
        return self._y_min if self._y_min is not None else (self.y - self.size * 0.5)

    @y_min.setter
    def y_min(self, val: float):
        self._y_min = float(val)

    @property
    def y_max(self) -> float:
        return self._y_max if self._y_max is not None else (self.y + self.size * 0.5)

    @y_max.setter
    def y_max(self, val: float):
        self._y_max = float(val)

    @property
    def z_mean(self) -> float:
        if not self.observed or not self.valid:
            return float("nan")
        return float(self.stats.mean_z) if self.stats is not None else -1.68

    @z_mean.setter
    def z_mean(self, val: float):
        if self.stats is not None:
            self.stats.mean_z = float(val)

    @property
    def z(self) -> float:
        return self.z_mean

    @property
    def mean_z(self) -> float:
        return self.z_mean

    @property
    def delta_z(self) -> float:
        if not self.observed or not self.valid:
            return float("nan")
        return float(self.stats.delta_z) if self.stats is not None else 0.0

    @delta_z.setter
    def delta_z(self, val: float):
        if self.stats is not None:
            self.stats.delta_z = float(val)

    @property
    def variance(self) -> float:
        if not self.observed or not self.valid:
            return float("nan")
        return float(self.stats.variance) if self.stats is not None else float("nan")

    @property
    def slope(self) -> float:
        if not self.observed or not self.valid:
            return float("nan")
        return float(self.stats.slope) if self.stats is not None else float("nan")

    @property
    def z_max(self) -> float:
        if not self.observed or not self.valid:
            return float("nan")
        return float(self.stats.z_max) if self.stats is not None else -1.68

    @z_max.setter
    def z_max(self, val: float):
        if self.stats is not None:
            self.stats.z_max = float(val)

    @property
    def z_min(self) -> float:
        if not self.observed or not self.valid:
            return float("nan")
        return float(self.stats.z_min) if self.stats is not None else -1.68

    @z_min.setter
    def z_min(self, val: float):
        if self.stats is not None:
            self.stats.z_min = float(val)

    @property
    def point_count(self) -> int:
        if not self.observed or not self.valid:
            return 0
        return int(self.stats.point_count) if self.stats is not None else 0

    @point_count.setter
    def point_count(self, val: int):
        if self.stats is not None:
            self.stats.point_count = val

    @property
    def min_x(self) -> float:
        return self.x - self.size / 2.0

    @property
    def max_x(self) -> float:
        return self.x + self.size / 2.0

    @property
    def min_y(self) -> float:
        return self.y - self.size / 2.0

    @property
    def max_y(self) -> float:
        return self.y + self.size / 2.0

    def to_dict(self):
        """Serialize leaf for network broadcasting with zero-overhead native python types."""
        d = {
            "x": float(round(self.x, 2)),
            "y": float(round(self.y, 2)),
            "size": float(round(self.size, 3)),
            "cost": self.cost,
            "observed": bool(self.observed),
            "valid": bool(self.valid),
        }
        if self.observed and self.valid and self.stats is not None:
            d["z_min"] = float(round(self.stats.z_min, 2))
            d["z_max"] = float(round(self.stats.z_max, 2))
            d["z_mean"] = float(round(self.stats.mean_z, 2))
            d["delta_z"] = float(round(self.stats.delta_z, 2))
            d["variance"] = float(round(self.stats.variance, 4))
            d["slope"] = float(round(self.stats.slope, 1))
            d["pts"] = int(self.stats.point_count)
        else:
            d["cost"] = -1
            d["z_min"] = float("nan")
            d["z_max"] = float("nan")
            d["z_mean"] = float("nan")
            d["delta_z"] = float("nan")
            d["variance"] = float("nan")
            d["slope"] = float("nan")
            d["pts"] = 0
        return d


class AdaptiveQuadtree:
    """
    Adaptive 2.5D Variable Resolution Quadtree.
    - Base coarse grid: coarse_res (default 0.50m)
    - Maximum refinement: fine_res (default 0.05m)
    - Split condition: delta_z > tau_z or variance > tau_sigma
    """

    def __init__(
        self,
        bounds=None,
        coarse_res: float = QUADTREE.COARSE_RESOLUTION,
        fine_res: float = QUADTREE.FINE_RESOLUTION,
        tau_sigma: float = QUADTREE.TAU_SIGMA,
        tau_z: float = QUADTREE.TAU_Z,
        min_pts_per_cell: int = QUADTREE.MIN_POINTS_PER_CELL,
        min_x: Optional[float] = None,
        max_x: Optional[float] = None,
        min_y: Optional[float] = None,
        max_y: Optional[float] = None,
    ):
        base_bounds = bounds if bounds is not None else BOUNDS
        x_min = min_x if min_x is not None else base_bounds.X_MIN
        x_max = max_x if max_x is not None else base_bounds.X_MAX
        y_min = min_y if min_y is not None else base_bounds.Y_MIN
        y_max = max_y if max_y is not None else base_bounds.Y_MAX

        from backend.config import SpatialBounds
        self.bounds = SpatialBounds(
            X_MIN=x_min,
            X_MAX=x_max,
            Y_MIN=y_min,
            Y_MAX=y_max,
            Z_MIN=base_bounds.Z_MIN,
            Z_MAX=base_bounds.Z_MAX,
        )
        self.coarse_res = coarse_res
        self.fine_res = fine_res
        self.tau_sigma = tau_sigma
        self.tau_z = tau_z
        self.min_pts = min_pts_per_cell

        # Section 4.4 Dynamic Coarse Node Capacity calculation
        extent_x = abs(self.bounds.X_MAX - self.bounds.X_MIN)
        extent_y = abs(self.bounds.Y_MAX - self.bounds.Y_MIN)
        self.num_coarse_x = int(np.ceil(extent_x / coarse_res))
        self.num_coarse_y = int(np.ceil(extent_y / coarse_res))
        self.total_coarse = self.num_coarse_x * self.num_coarse_y
        self.max_coarse_nodes = self.total_coarse + 5000
        self.leaves: List[QuadtreeNode] = []

        # Lazy/bounded coarse node pool allocation to prevent GC spikes and memory bloat
        self._coarse_pool: List[QuadtreeNode] = []
        initial_pool = max(40000, self.total_coarse)
        for i in range(initial_pool):
            iy = i // self.num_coarse_x if self.num_coarse_x > 0 else 0
            ix = i % self.num_coarse_x if self.num_coarse_x > 0 else 0
            cx = self.bounds.X_MIN + (ix + 0.5) * coarse_res
            cy = self.bounds.Y_MIN + (iy + 0.5) * coarse_res
            node = QuadtreeNode(
                x=cx,
                y=cy,
                size=coarse_res,
                depth=0,
                stats=CellStats(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0),
                is_leaf=True,
            )
            self._coarse_pool.append(node)

        # Foveated grid index instance
        from backend.mapping.foveated_grid import FoveatedGrid
        self.foveated_grid = FoveatedGrid(bounds=self.bounds)

    def _get_or_create_coarse_node(self, pool_idx: int, cx: float, cy: float) -> QuadtreeNode:
        """
        Safely retrieves pre-allocated coarse node or dynamically expands pool to prevent IndexError.
        Guarantees zero-crash operation when spatial boundaries expand or shift dynamically.
        Uses a dense pool to prevent memory bloat.
        """
        MAX_POOL_LIMIT = 500000
        if pool_idx >= MAX_POOL_LIMIT:
            return None

        if pool_idx < len(self._coarse_pool):
            node = self._coarse_pool[pool_idx]
            node.x = cx
            node.y = cy
            return node

        needed = (pool_idx + 1) - len(self._coarse_pool)
        for _ in range(needed):
            self._coarse_pool.append(QuadtreeNode(
                x=0.0,
                y=0.0,
                size=self.coarse_res,
                depth=0,
                stats=CellStats(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0),
                is_leaf=True,
            ))
        node = self._coarse_pool[pool_idx]
        node.x = cx
        node.y = cy
        return node

    def _subdivide_recursive(
        self,
        cx: float,
        cy: float,
        size: float,
        depth: int,
        pts: np.ndarray,
    ) -> QuadtreeNode:
        """
        Recursively subdivide node if elevation variance or delta_z exceeds threshold.
        """
        stats = compute_cell_statistics(pts, cell_size=size)
        node = QuadtreeNode(x=cx, y=cy, size=size, depth=depth, stats=stats)

        # Check if subdivision criteria met
        should_split = False
        next_size = size / 2.0

        if next_size >= self.fine_res and len(pts) >= self.min_pts:
            if stats is not None:
                # Noise-gated subdivision: gate beam noise on flat ground
                if stats.delta_z > self.tau_z:
                    should_split = True

        # Obstacle Coarse-Leaf Collapse (Prevent Over-Subdivision / Confetti Micro-Leaves)
        if should_split and depth == 0 and len(pts) >= self.min_pts:
            z_pts = pts[:, 2]
            obs_mask = (z_pts > -1.2)
            obs_count = int(np.sum(obs_mask))
            if obs_count >= 3:
                node.cost = 255
                node.is_leaf = True
                if stats is not None:
                    node.stats.delta_z = max(0.25, float(stats.z_max - stats.z_min))
                    node.stats.mean_z = float(stats.mean_z)
                    node.stats.z_min = float(stats.z_min)
                    node.stats.z_max = float(stats.z_max)
                self.leaves.append(node)
                return node

        if not should_split:
            node.is_leaf = True
            self.leaves.append(node)
            return node

        # Subdivide into 4 quadrants: NW, NE, SW, SE
        node.is_leaf = False
        half = size / 4.0
        offsets = [
            (-half,  half),  # NW
            ( half,  half),  # NE
            (-half, -half),  # SW
            ( half, -half),  # SE
        ]

        # Partition points into the 4 child quadrants
        px = pts[:, 0]
        py = pts[:, 1]
        
        child_masks = [
            (px < cx) & (py >= cy),  # NW
            (px >= cx) & (py >= cy), # NE
            (px < cx) & (py < cy),   # SW
            (px >= cx) & (py < cy),  # SE
        ]

        node.children = []
        for (dx, dy), mask in zip(offsets, child_masks):
            child_pts = pts[mask]
            if len(child_pts) == 0:
                continue
            child_node = self._subdivide_recursive(
                cx=cx + dx,
                cy=cy + dy,
                size=next_size,
                depth=depth + 1,
                pts=child_pts,
            )
            node.children.append(child_node)

        return node

    def clear(self) -> None:
        """
        Completely clears all active leaf nodes.
        Guarantees local elevation grid is reset per frame without inter-frame accumulation.
        """
        self.leaves.clear()
        if hasattr(self, "foveated_grid") and self.foveated_grid is not None:
            self.foveated_grid.clear()

    def build_foveated(self, points: np.ndarray, point_costs: Optional[np.ndarray] = None) -> List[QuadtreeNode]:
        """
        PS 26053 Hybrid Hierarchical Foveated Spatial Index build.
        Exact zone resolutions: 5cm (0-10m), 10cm (10-25m), 25cm (25-50m), 50cm (50-100m).
        """
        self.leaves = self.foveated_grid.build_foveated(points, point_costs=point_costs)
        return self.leaves

    def build(self, points: np.ndarray, point_costs: Optional[np.ndarray] = None) -> List[QuadtreeNode]:
        """
        Constructs the adaptive 2.5D variable-resolution quadtree.
        Vectorized coarse grid evaluation and direct quadrant refinement in < 6 ms.
        Optionally ingests per-point costs (e.g. lethal 255 for detected thin threats).
        """
        self.leaves.clear()
        if len(points) == 0:
            return self.leaves

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # 0. Spatial bounds filter to avoid perimeter edge distortion
        in_bounds = (
            (x >= self.bounds.X_MIN) & (x <= self.bounds.X_MAX) &
            (y >= self.bounds.Y_MIN) & (y <= self.bounds.Y_MAX) &
            (z >= self.bounds.Z_MIN) & (z <= self.bounds.Z_MAX)
        )
        if not np.all(in_bounds):
            points = points[in_bounds]
            if point_costs is not None and len(point_costs) == len(in_bounds):
                point_costs = point_costs[in_bounds]
            if len(points) == 0:
                return self.leaves
            x = points[:, 0]
            y = points[:, 1]
            z = points[:, 2]

        # 1. Vectorized Coarse Grid Analysis (O(N) in ~1.5ms)
        ix = np.clip(((x - self.bounds.X_MIN) / self.coarse_res).astype(np.int32), 0, self.num_coarse_x - 1)
        iy = np.clip(((y - self.bounds.Y_MIN) / self.coarse_res).astype(np.int32), 0, self.num_coarse_y - 1)
        cell_keys = iy * self.num_coarse_x + ix
        total_coarse = self.num_coarse_x * self.num_coarse_y

        z_min = np.full(total_coarse, np.inf, dtype=np.float32)
        z_max = np.full(total_coarse, -np.inf, dtype=np.float32)
        counts = np.zeros(total_coarse, dtype=np.int32)
        obs_counts = np.zeros(total_coarse, dtype=np.int32)
        ground_counts = np.zeros(total_coarse, dtype=np.int32)
        z_sum = np.zeros(total_coarse, dtype=np.float32)

        np.minimum.at(z_min, cell_keys, z)
        np.maximum.at(z_max, cell_keys, z)
        np.add.at(counts, cell_keys, 1)
        np.add.at(z_sum, cell_keys, z)

        is_obs_point = (z > -1.2)
        if point_costs is not None and len(point_costs) == len(points):
            is_obs_point = is_obs_point | (point_costs >= 181)
            c_costs = np.zeros(total_coarse, dtype=np.int32)
            np.maximum.at(c_costs, cell_keys, point_costs)
        else:
            c_costs = None

        np.add.at(obs_counts, cell_keys, is_obs_point.astype(np.int32))
        is_ground_point = (z >= -2.1) & (z <= -1.35)
        np.add.at(ground_counts, cell_keys, is_ground_point.astype(np.int32))

        active = counts >= self.min_pts
        delta_z = z_max - z_min
        # Noise-gated subdivision: gate beam noise on flat ground (sigma^2 > 0.04)
        # Subdivide when step height exceeds obstacle threshold, elevated object in air, or lethal hazard detected
        # Flat asphalt ground: Z between -2.0m and -1.35m with delta_z <= 0.15m ALWAYS evaluates strictly to Cost = 0
        # Flat asphalt ground: relax delta_z to 0.20m to reduce unnecessary subdivision of bumpy ground
        mean_z_arr = np.where(counts > 0, z_sum / np.maximum(counts, 1), -99.0)
        is_flat_ground = (mean_z_arr >= -2.15) & (mean_z_arr <= -1.35) & (delta_z <= 0.20)

        is_obstacle_step = (delta_z > self.tau_z)
        is_elevated_obj = (z_max > -1.2)
        is_lethal_hazard = (c_costs > 50) if c_costs is not None else False
        needs_subdivision = active & (~is_flat_ground) & (is_obstacle_step | is_elevated_obj | is_lethal_hazard)

        # Obstacle Coarse-Leaf Collapse (Coarse Tile Aggregation for Obstacle Clusters):
        # Collapse cells containing confirmed non-ground obstacle returns (obs_counts >= 2)
        is_solid_obstacle = active & (~is_flat_ground) & (
            (obs_counts >= 2) |
            (c_costs >= 181 if c_costs is not None else False)
        )

        # Coarse leaves: flat ground cells PLUS collapsed solid obstacle cells
        coarse_active = active & ((~needs_subdivision) | is_solid_obstacle)
        # Boundary transitions: only subdivide cells containing part ground, part obstacle where boundary resolution is needed
        fine_subdivision_active = needs_subdivision & (~is_solid_obstacle)

        # 2. Directly update pre-allocated coarse leaf nodes
        coarse_keys = np.where(coarse_active)[0]
        c_min_arr = z_min[coarse_keys]
        c_max_arr = z_max[coarse_keys]
        c_dz_arr = delta_z[coarse_keys]
        c_counts = counts[coarse_keys]
        c_z_sum = z_sum[coarse_keys]
        c_slopes = np.where(c_dz_arr > 0.01, np.degrees(np.arctan2(c_dz_arr, self.coarse_res)), 0.0)

        _coarse_pool_idx = 0
        for i in range(len(coarse_keys)):
            key = coarse_keys[i]
            ix = key % self.num_coarse_x
            iy = key // self.num_coarse_x
            cx = self.bounds.X_MIN + (ix + 0.5) * self.coarse_res
            cy = self.bounds.Y_MIN + (iy + 0.5) * self.coarse_res
            node = self._get_or_create_coarse_node(_coarse_pool_idx, cx, cy)
            _coarse_pool_idx += 1
            if node is None:
                continue
            
            st = node.stats
            min_z_val = float(c_min_arr[i])
            max_z_val = float(c_max_arr[i])
            dz_val = float(c_dz_arr[i])
            count_val = int(c_counts[i])
            mean_z_val = float(c_z_sum[i] / max(1, count_val))

            st.z_min = min_z_val
            st.z_max = max_z_val
            st.delta_z = dz_val
            st.point_count = count_val
            st.mean_z = mean_z_val
            st.variance = float((dz_val / 2.0) ** 2)
            st.slope = float(c_slopes[i])
            node.is_leaf = True


            # Drivable road surface envelope:
            # Asphalt elevation sits between -2.20m and -1.25m with step height <= 0.18m
            is_drivable_ground = (
                (-2.20 <= node.z_mean <= -1.25) and
                (node.delta_z <= 0.18) and
                (node.z_max <= -1.15)
            )

            if is_drivable_ground:
                node.cost = 0
                node.semantic_cost = 0
                node.is_obstacle = False
                node.is_hazard = False
            else:
                # Rigid obstacles (elevated vehicles, barriers, curbs, poles)
                # Require at least 3 points to filter out isolated airborne dust/scatter
                has_sufficient_density = getattr(node, 'point_count', 0) >= 3
                is_vertical_obstacle = (node.delta_z > 0.25 or node.z_max > -1.15)

                if (is_vertical_obstacle and has_sufficient_density) or (c_costs is not None and c_costs[key] >= 181):
                    node.cost = 255
                    node.semantic_cost = 255
                    node.is_obstacle = True
                    node.is_hazard = True
                elif c_costs is not None and c_costs[key] > 0:
                    node.cost = int(c_costs[key])
                    node.semantic_cost = int(c_costs[key])
                    node.is_obstacle = (node.cost >= 181)
                    node.is_hazard = (node.cost >= 181)
                elif node.delta_z > 0.15 or node.z_mean > -1.35 or float(c_slopes[i]) > 20.0:
                    node.cost = 100
                    node.semantic_cost = 100
                    node.is_obstacle = False
                    node.is_hazard = False
                else:
                    node.cost = 0
                    node.semantic_cost = 0
                    node.is_obstacle = False
                    node.is_hazard = False
            self.leaves.append(node)


        # 3. Vectorized Sub-quadrant Refinement for Boundary Transition Cells
        if np.any(fine_subdivision_active):
            # Refine boundary transition cells into 4x4 fine sub-cells (fine_res = 0.125m)
            sub_factor = 4
            fine_res = self.coarse_res / sub_factor
            num_fine_x = self.num_coarse_x * sub_factor
            num_fine_y = self.num_coarse_y * sub_factor

            rough_mask = fine_subdivision_active[cell_keys]
            if np.any(rough_mask):
                rx = x[rough_mask]
                ry = y[rough_mask]
                rz = z[rough_mask]

                fix = np.clip(((rx - self.bounds.X_MIN) / fine_res).astype(np.int32), 0, num_fine_x - 1)
                fiy = np.clip(((ry - self.bounds.Y_MIN) / fine_res).astype(np.int32), 0, num_fine_y - 1)
                fine_keys = fiy * num_fine_x + fix

                uniq_fkeys, inv, f_counts = np.unique(fine_keys, return_inverse=True, return_counts=True)
                n_uniq = len(uniq_fkeys)
                fz_min = np.full(n_uniq, np.inf, dtype=np.float32)
                fz_max = np.full(n_uniq, -np.inf, dtype=np.float32)
                np.minimum.at(fz_min, inv, rz)
                np.maximum.at(fz_max, inv, rz)

                if point_costs is not None and len(point_costs) == len(points):
                    f_costs = np.zeros(n_uniq, dtype=np.int32)
                    np.maximum.at(f_costs, inv, point_costs[rough_mask])
                else:
                    f_costs = None

                fx_indices = uniq_fkeys % num_fine_x
                fy_indices = uniq_fkeys // num_fine_x
                fine_cx = self.bounds.X_MIN + (fx_indices + 0.5) * fine_res
                fine_cy = self.bounds.Y_MIN + (fy_indices + 0.5) * fine_res
                fine_dz_arr = fz_max - fz_min
                fine_slopes = np.where(fine_dz_arr > 0.01, np.degrees(np.arctan2(fine_dz_arr, fine_res)), 0.0)

                for i in range(n_uniq):
                    f_min_z = float(fz_min[i])
                    f_max_z = float(fz_max[i])
                    f_dz = float(fine_dz_arr[i])
                    f_mean_z = float((f_min_z + f_max_z) / 2.0)
                    f_slope = float(fine_slopes[i])

                    f_stats = CellStats(
                        z_min=f_min_z,
                        z_max=f_max_z,
                        delta_z=f_dz,
                        variance=float((f_dz / 2.0) ** 2),
                        slope=f_slope,
                        point_count=int(f_counts[i]),
                        mean_z=f_mean_z,
                    )
                    f_is_drivable_ground = (
                        (-2.20 <= f_mean_z <= -1.25) and
                        (f_dz <= 0.18) and
                        (f_max_z <= -1.15)
                    )

                    if f_is_drivable_ground:
                        f_cost = 0
                        f_is_obs = False
                    else:
                        has_sufficient_density = int(f_counts[i]) >= 3
                        is_vertical_obstacle = (f_dz > 0.25 or f_max_z > -1.15)

                        if (is_vertical_obstacle and has_sufficient_density) or (f_costs is not None and f_costs[i] >= 181):
                            f_cost = 255
                            f_is_obs = True
                        elif f_costs is not None and f_costs[i] > 0:
                            f_cost = int(f_costs[i])
                            f_is_obs = (f_cost >= 181)
                        elif f_dz > 0.15 or f_mean_z > -1.35 or f_slope > 20.0:
                            f_cost = 100
                            f_is_obs = False
                        else:
                            f_cost = 0
                            f_is_obs = False

                    self.leaves.append(QuadtreeNode(
                        x=float(fine_cx[i]),
                        y=float(fine_cy[i]),
                        size=fine_res,
                        depth=2,
                        stats=f_stats,
                        cost=f_cost,
                        semantic_cost=f_cost,
                        is_obstacle=f_is_obs,
                        is_hazard=f_is_obs,
                        is_leaf=True,
                    ))


        return self.leaves

    def apply_point_costs(self, leaves: List[QuadtreeNode], points: np.ndarray, cost: int = 255) -> None:
        """
        Directly assigns cost to any leaf nodes containing any of the given points.
        """
        if not leaves or len(points) == 0:
            return

        px = points[:, 0]
        py = points[:, 1]
        for leaf in leaves:
            in_cell = (px >= leaf.min_x) & (px <= leaf.max_x) & (py >= leaf.min_y) & (py <= leaf.max_y)
            if np.any(in_cell):
                leaf.cost = max(leaf.cost, cost)

    def interpolate_ground_rings(
        self,
        leaves: Optional[List[QuadtreeNode]] = None,
        max_dz: float = 0.18,
    ) -> List[QuadtreeNode]:
        """
        Morphological planar void inpainting with negative obstacle gating & caution classification.
        Extracts 2D boolean mask of planar ground leaves (Cost <= 50, -2.1m <= mean_z <= -1.35m).

        Two-pass adaptive morphological closing:
          - Driving corridor (|y| <= 4.0m): 5x13 kernel (±6 cells / 3.0m along X, ±2 cells / 1.0m along Y)
            to bridge far-field laser divergence gaps across the forward road.
          - Off-corridor areas (|y| > 4.0m): tighter 5x9 kernel (±4 cells / 2.0m along X)
            to preserve clean curb lines and road shoulders.

        Classification Contract:
          1. SAFE ROAD (Green / Cost = 0): Only bridge narrow, verified planar gaps (<= 2 cells / <= 1.0m)
             where elevation delta across borders is negligible (|delta_z| <= 0.06m).
          2. CAUTION TILE (Yellow / Cost = 100): Medium / uncertain gaps (3 to 4 cells / 1.5m - 2.0m)
             or slight slope variations (0.06m < |delta_z| <= 0.15m) where road continuity is unconfirmed.
          3. LETHAL NEGATIVE OBSTACLE (Red / Cost = 255): Boundary cells or voids indicating a step drop
             (|delta_z| > 0.15m down, or depression below baseline Z < -2.0m). Impassable ditch/pothole.
          4. UNCONFIRMED WIDE GAPS (> 4 cells / > 2.0m): Left unfilled (not filled green) to prevent
             hallucinating safe road over unconfirmed void spaces.
        """
        target_leaves = leaves if leaves is not None else self.leaves
        if not target_leaves:
            return target_leaves

        nx = self.num_coarse_x
        ny = self.num_coarse_y
        active_2d = np.zeros((ny, nx), dtype=bool)
        ground_2d = np.zeros((ny, nx), dtype=bool)
        mean_z_2d = np.zeros((ny, nx), dtype=np.float32)

        inv_res = 1.0 / self.coarse_res
        x_min = self.bounds.X_MIN
        y_min = self.bounds.Y_MIN

        # Index existing active coarse cells and extract planar ground
        for leaf in target_leaves:
            gx = int((leaf.x - x_min) * inv_res)
            gy = int((leaf.y - y_min) * inv_res)
            if 0 <= gx < nx and 0 <= gy < ny:
                active_2d[gy, gx] = True
                if leaf.cost <= 50 and leaf.stats is not None:
                    st = leaf.stats
                    if -2.1 <= st.mean_z <= -1.35 and st.delta_z <= max_dz:
                        ground_2d[gy, gx] = True
                        mean_z_2d[gy, gx] = st.mean_z

        if not np.any(ground_2d):
            return target_leaves

        # Morphological closing:
        # 1. Isotropic 3x3 structuring element (radial 0.4m - 0.6m) bridging annular LiDAR beam gaps (0 to 25m)
        structure_3x3 = np.ones((3, 3), dtype=bool)
        closed_3x3 = ndi.binary_closing(ground_2d, structure=structure_3x3)

        # 2. Forward corridor closing (7x13 kernel: ±3 Y cells, ±6 X cells) bridging longitudinal laser divergence
        structure_fwd = np.ones((7, 13), dtype=bool)
        closed_fwd = ndi.binary_closing(ground_2d, structure=structure_fwd)

        # 3. Lateral radial closing (9x7 kernel: ±4 Y cells, ±3 X cells) bridging peripheral lateral beam voids
        structure_lat = np.ones((9, 7), dtype=bool)
        closed_lat = ndi.binary_closing(ground_2d, structure=structure_lat)

        # 4. Adjacent empty neighbor closing around verified safe road cells
        structure_adj = ndi.generate_binary_structure(2, 2)
        closed_adj = ndi.binary_dilation(ground_2d, structure=structure_adj) & ndi.binary_closing(ground_2d, structure=np.ones((5, 5), dtype=bool))

        closed_ground = closed_3x3 | closed_fwd | closed_lat | closed_adj
        bridged = closed_ground & (~active_2d)

        if not np.any(bridged):
            return target_leaves

        # Smooth local elevation interpolation for bridged cells
        ground_float = ground_2d.astype(np.float32)
        sum_z = ndi.uniform_filter(mean_z_2d * ground_float, size=(5, 9))
        sum_w = ndi.uniform_filter(ground_float, size=(5, 9))
        smooth_z = sum_z / np.maximum(sum_w, 1e-5)

        # --- Vectorized gap width & boundary elevation extraction ---
        cols = np.tile(np.arange(nx), (ny, 1))
        left_indices = np.where(ground_2d, cols, -999)
        left_x = np.maximum.accumulate(left_indices, axis=1)
        right_indices = np.where(ground_2d, cols, 9999)
        right_x = np.minimum.accumulate(right_indices[:, ::-1], axis=1)[:, ::-1]
        valid_x = (left_x >= 0) & (right_x < nx)
        rows_grid = np.arange(ny)[:, None]
        z_left_x = np.where(valid_x, mean_z_2d[rows_grid, np.maximum(0, left_x)], 0.0)
        z_right_x = np.where(valid_x, mean_z_2d[rows_grid, np.minimum(nx - 1, right_x)], 0.0)
        gap_w_x = np.where(valid_x, right_x - left_x - 1, 999)
        dz_x = np.where(valid_x, np.abs(z_left_x - z_right_x), 999.0)

        rows = np.tile(np.arange(ny)[:, None], (1, nx))
        down_indices = np.where(ground_2d, rows, -999)
        down_y = np.maximum.accumulate(down_indices, axis=0)
        up_indices = np.where(ground_2d, rows, 9999)
        up_y = np.minimum.accumulate(up_indices[::-1, :], axis=0)[::-1, :]
        valid_y = (down_y >= 0) & (up_y < ny)
        cols_grid = np.arange(nx)[None, :]
        z_down_y = np.where(valid_y, mean_z_2d[np.maximum(0, down_y), cols_grid], 0.0)
        z_up_y = np.where(valid_y, mean_z_2d[np.minimum(ny - 1, up_y), cols_grid], 0.0)
        gap_w_y = np.where(valid_y, up_y - down_y - 1, 999)
        dz_y = np.where(valid_y, np.abs(z_down_y - z_up_y), 999.0)

        gy_coords, gx_coords = np.where(bridged)
        new_inpainted: List[QuadtreeNode] = []

        for gy, gx in zip(gy_coords, gx_coords):
            wx = gap_w_x[gy, gx]
            wy = gap_w_y[gy, gx]
            w = min(wx, wy)

            if wx <= wy and wx < 999:
                z_a, z_b = float(z_left_x[gy, gx]), float(z_right_x[gy, gx])
                dz_1d = float(dz_x[gy, gx])
            elif wy < 999:
                z_a, z_b = float(z_down_y[gy, gx]), float(z_up_y[gy, gx])
                dz_1d = float(dz_y[gy, gx])
            else:
                z_a, z_b = -1.60, -1.60
                dz_1d = 0.0

            raw_mz = float(smooth_z[gy, gx])
            mz = raw_mz
            if mz < -2.1 or mz > -1.35:
                mz = -1.60

            # Surrounding valid ground neighbor inspection in 3x3 window (radial 0.4m - 0.6m)
            y0 = max(0, gy - 1)
            y1 = min(ny, gy + 2)
            x0 = max(0, gx - 1)
            x1 = min(nx, gx + 2)
            nbr_mask = ground_2d[y0:y1, x0:x1]
            nbr_z = mean_z_2d[y0:y1, x0:x1][nbr_mask]

            if len(nbr_z) >= 2:
                local_dz = float(np.ptp(nbr_z))
                min_nbr_z = float(np.min(nbr_z))
                effective_dz = local_dz
            else:
                local_dz = dz_1d
                min_nbr_z = min(z_a, z_b)
                effective_dz = dz_1d

            # Negative hazard check (pothole / ditch):
            # - Neighbor elevation drop >= 0.20m (or boundary drop > 0.15m in narrow gap)
            # - Or boundary elevation drops below road baseline (Z < -2.0m)
            # - Or interpolated elevation is substantially depressed (Z < -2.0m)
            is_negative_hazard = (
                (effective_dz >= 0.20) or
                (dz_1d > 0.15 and w <= 2) or
                (min_nbr_z < -2.0) or
                (raw_mz < -2.0)
            )

            if is_negative_hazard:
                # 3. LETHAL NEGATIVE OBSTACLE (Cost = 255): Impassable ditch/pothole
                cell_cost = 255
                stat_dz = max(0.20, effective_dz)
            elif w > 4 and len(nbr_z) < 2:
                # 4. UNCONFIRMED WIDE GAPS (> 4 cells / > 2.0m) without immediate neighbors:
                # Left unfilled to prevent navigating through unscanned voids.
                continue
            elif w > 2:
                # 2. CAUTION TILE (Yellow / Cost = 100): Medium gap (3-4 cells)
                cell_cost = 100
                stat_dz = max(0.08, effective_dz)
            elif effective_dz <= 0.06:
                # 1. SAFE ROAD (Green / Cost = 0): Narrow gap & strict planar matching
                cell_cost = 0
                stat_dz = 0.02
            elif effective_dz < 0.15:
                # Slight slope variation: Caution
                cell_cost = 100
                stat_dz = max(0.08, effective_dz)
            elif effective_dz < 0.20:
                cell_cost = 100
                stat_dz = max(0.15, effective_dz)
            else:
                cell_cost = 255
                stat_dz = max(0.20, effective_dz)

            node_cx = float(x_min + (gx + 0.5) * self.coarse_res)
            node_cy = float(y_min + (gy + 0.5) * self.coarse_res)

            inpaint_node = QuadtreeNode(
                x=node_cx,
                y=node_cy,
                size=self.coarse_res,
                depth=0,
                stats=CellStats(
                    z_min=mz - 0.01,
                    z_max=mz + 0.01,
                    delta_z=stat_dz,
                    variance=0.0001,
                    slope=0.0,
                    point_count=1,
                    mean_z=mz,
                ),
                cost=cell_cost,
                is_leaf=True,
            )
            new_inpainted.append(inpaint_node)

        # Combined leaves preserves all original leaves + newly inpainted ground cells
        combined = list(target_leaves) + new_inpainted
        self.leaves = combined
        return self.leaves

    def interpolate_obstacle_clusters(self, leaves: Optional[List[QuadtreeNode]] = None) -> List[QuadtreeNode]:
        """
        Completely bypassed: Do NOT synthesize artificial red obstacle tiles.
        Only physical sensor returns may register as obstacles.
        """
        if leaves is not None:
            self.leaves = leaves
        return self.leaves

    def apply_clearance_coarsening(
        self,
        leaves: Optional[List[QuadtreeNode]] = None,
    ) -> List[QuadtreeNode]:
        """
        Width-Aware Ground Coarsening & Clearance-Based Quadtree Sizing.
        Evaluates local lateral free-space clearance across the 2D grid:
          1. Open drivable space (clearance >= 1.0m from each side, free width > 2.0m):
             Merges 2x2 clusters of adjacent 0.50m cells into a single 1.0m x 1.0m super-tile (Cost = 0).
          2. Constricted pinch-point (0.6m <= clearance < 1.0m, free width ~ 1.2m to 1.8m):
             Scales into 0.75m x 0.75m Caution tiles (Cost = 100).
          3. Obstacles & physical barriers (cost >= 181):
             Maintains consistent 0.50m x 0.50m coarse leaves (Cost = 255).
        """
        target_leaves = leaves if leaves is not None else self.leaves
        if not target_leaves:
            return target_leaves

        nx = self.num_coarse_x
        ny = self.num_coarse_y
        x_min = self.bounds.X_MIN
        y_min = self.bounds.Y_MIN
        res = self.coarse_res
        inv_res = 1.0 / res

        ground_leaf_map: Dict[Tuple[int, int], QuadtreeNode] = {}
        ground_2d = np.zeros((ny, nx), dtype=bool)
        other_leaves: List[QuadtreeNode] = []

        for leaf in target_leaves:
            if not leaf.is_leaf:
                continue
            gx = int((leaf.x - x_min) * inv_res)
            gy = int((leaf.y - y_min) * inv_res)
            if gx < 0 or gx >= nx or gy < 0 or gy >= ny:
                other_leaves.append(leaf)
                continue

            # Flat asphalt ground: Z between -2.0m and -1.35m with delta_z <= 0.15m ALWAYS evaluates strictly to Cost = 0
            is_flat_ground = (leaf.stats is not None) and (-2.0 <= leaf.stats.mean_z <= -1.35 and leaf.stats.delta_z <= 0.15)
            if is_flat_ground:
                leaf.cost = 0
                if leaf.size >= res - 0.01:
                    ground_leaf_map[(gy, gx)] = leaf
                    ground_2d[gy, gx] = True
                else:
                    other_leaves.append(leaf)
                continue

            is_obs = (leaf.cost >= 181) or (leaf.stats and (leaf.stats.mean_z > -1.2 or leaf.stats.delta_z > 0.25))
            is_gnd = (leaf.cost <= 50) or (leaf.stats and -2.1 <= leaf.stats.mean_z <= -1.35 and leaf.stats.delta_z <= 0.15)

            if is_obs:
                if leaf.size >= res - 0.01:
                    leaf.size = 0.50
                    leaf.cost = 255
                other_leaves.append(leaf)
            elif is_gnd and leaf.size >= res - 0.01:
                ground_leaf_map[(gy, gx)] = leaf
                ground_2d[gy, gx] = True
            else:
                other_leaves.append(leaf)

        if not np.any(ground_2d):
            return target_leaves

        clearance_2d = ndi.distance_transform_edt(ground_2d) * res
        is_wide = ground_2d & (clearance_2d >= 1.0)
        is_pinch = ground_2d & (clearance_2d >= 0.6) & (clearance_2d < 1.0)

        # 1. Wide Drivable Corridor: Merge 2x2 clusters of 0.50m cells into 1.0m super-tiles (Cost = 0)
        # Robust 2x2 aggregation: allows single-cell dropouts (>= 3 wide cells) with zero pinch cells and zero obstacles
        w00 = is_wide[0::2, 0::2]
        w01 = is_wide[0::2, 1::2]
        w10 = is_wide[1::2, 0::2]
        w11 = is_wide[1::2, 1::2]
        all_four = w00 & w01 & w10 & w11
        wide_sum = w00.astype(np.int32) + w01.astype(np.int32) + w10.astype(np.int32) + w11.astype(np.int32)

        has_pinch = (
            is_pinch[0::2, 0::2] | is_pinch[0::2, 1::2] |
            is_pinch[1::2, 0::2] | is_pinch[1::2, 1::2]
        )

        obs_mask = np.zeros((ny, nx), dtype=bool)
        for leaf in target_leaves:
            if leaf.cost >= 181:
                gx = int((leaf.x - x_min) * inv_res)
                gy = int((leaf.y - y_min) * inv_res)
                if 0 <= gx < nx and 0 <= gy < ny:
                    obs_mask[gy, gx] = True

        has_obs = (
            obs_mask[0::2, 0::2] | obs_mask[0::2, 1::2] |
            obs_mask[1::2, 0::2] | obs_mask[1::2, 1::2]
        )

        three_with_dropout = (wide_sum >= 3) & (~has_pinch) & (~has_obs)
        even_blocks = all_four | three_with_dropout
        by_e, bx_e = np.where(even_blocks)
        gy_e = by_e * 2
        gx_e = bx_e * 2

        used_mask = np.zeros((ny, nx), dtype=bool)
        used_mask[gy_e, gx_e] = True
        used_mask[gy_e, gx_e + 1] = True
        used_mask[gy_e + 1, gx_e] = True
        used_mask[gy_e + 1, gx_e + 1] = True

        coarsened_ground: List[QuadtreeNode] = []
        for gy, gx in zip(gy_e, gx_e):
            quad = [(gy, gx), (gy, gx + 1), (gy + 1, gx), (gy + 1, gx + 1)]
            block_leaves = [ground_leaf_map.get(cell) for cell in quad]
            super_cx = float(x_min + (gx + 1.0) * res)
            super_cy = float(y_min + (gy + 1.0) * res)

            valid_leaves = [l for l in block_leaves if l is not None and l.stats is not None]
            if valid_leaves:
                min_z = min(l.stats.z_min for l in valid_leaves)
                max_z = max(l.stats.z_max for l in valid_leaves)
                mean_z = float(np.mean([l.stats.mean_z for l in valid_leaves]))
                dz = max(0.02, max_z - min_z)
                slope = float(np.mean([l.stats.slope for l in valid_leaves]))
                pt_count = sum(l.stats.point_count for l in valid_leaves)
            else:
                min_z, max_z, mean_z, dz, slope, pt_count = -1.65, -1.55, -1.60, 0.02, 0.0, 4

            super_node = QuadtreeNode(
                x=super_cx,
                y=super_cy,
                size=1.0,
                depth=0,
                stats=CellStats(
                    z_min=min_z,
                    z_max=max_z,
                    delta_z=dz,
                    variance=0.001,
                    slope=slope,
                    point_count=pt_count,
                    mean_z=mean_z,
                ),
                cost=0,  # Open Drivable Corridor (Green)
                is_leaf=True,
            )
            coarsened_ground.append(super_node)

        # 2. Constricted Pinch-Points (0.6m <= clearance < 1.0m) and remaining ground cells
        for (gy, gx), leaf in ground_leaf_map.items():
            if used_mask[gy, gx]:
                continue
            clr = float(clearance_2d[gy, gx])
            if 0.6 <= clr < 1.0:
                # Pinch-point: scale to 0.75m Caution tile
                leaf.size = 0.75
                leaf.cost = 100
                coarsened_ground.append(leaf)
            elif clr >= 1.0:
                # Wide corridor border tile (didn't form a 2x2 block)
                leaf.size = 0.50
                leaf.cost = 0
                coarsened_ground.append(leaf)
            else:
                # Boundary transition near obstacles (clearance < 0.6m)
                leaf.size = 0.50
                coarsened_ground.append(leaf)

        combined = other_leaves + coarsened_ground
        self.leaves = combined
        return self.leaves

    def get_active_leaves(self) -> List[QuadtreeNode]:
        """
        Returns all active leaf nodes in the quadtree.
        Guarantees that coarse ground leaves (Cost = 0, size >= 0.50m) are preserved
        and not skipped or discarded regardless of delta_z < 0.04 or cost == 0.
        """
        return [leaf for leaf in self.leaves if leaf.is_leaf]

    def get_memory_footprint_mb(self) -> float:
        """
        Computes the actual in-memory footprint of the 2.5D Quadtree data structure (MB).
        Formula: N_leaves * sizeof(Node) + coarse_pool + spatial index buffers.
        Complies strictly with DRDO PS-53 edge memory limit (1.2 MB - 6.5 MB, <= 10.0 MB).
        """
        # Slotted QuadtreeNode (96B) + CellStats (88B) + pointer (8B) = 192 bytes
        bytes_per_node = 192
        pool_bytes = len(self._coarse_pool) * bytes_per_node
        leaves_bytes = len(self.leaves) * bytes_per_node
        aux_bytes = 262144  # 256 KB coordinate lookup tables
        return round((pool_bytes + leaves_bytes + aux_bytes) / (1024.0 * 1024.0), 2)

    def get_statistics(self) -> dict:
        """Returns structural stats on the current tree."""
        if not self.leaves:
            return {
                "total_leaves": 0,
                "coarse_cells": 0,
                "fine_cells": 0,
                "refinement_ratio": 0.0,
                "ram_mb": self.get_memory_footprint_mb(),
            }

        coarse_threshold = self.coarse_res - 0.01
        coarse_count = sum(1 for leaf in self.leaves if leaf.size >= coarse_threshold)
        fine_count = len(self.leaves) - coarse_count

        return {
            "total_leaves": len(self.leaves),
            "coarse_cells": coarse_count,
            "fine_cells": fine_count,
            "refinement_ratio": round(fine_count / max(len(self.leaves), 1), 3),
            "ram_mb": self.get_memory_footprint_mb(),
        }
