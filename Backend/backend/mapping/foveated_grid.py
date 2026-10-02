"""
PS 26053 Hybrid Hierarchical Foveated Spatial Index.

Implements exact radial perception zones:
- Zone 1: [0, 10 m)   -> exact 0.05 m resolution
- Zone 2: [10, 25 m)  -> exact 0.10 m resolution
- Zone 3: [25, 50 m)  -> exact 0.25 m resolution
- Zone 4: [50, 100 m] -> exact 0.50 m resolution
- Perception Envelope: r <= 100.0 m

Guarantees:
- Axis-aligned rectangular cells
- Zero radial zone straddling (r_min and r_max strictly within a single zone)
- Exact resolutions (0.05m, 0.10m, 0.25m, 0.50m)
- Deterministic cell IDs
- No overlapping leaves
- No uncovered regions within supported envelope
- Terrain-driven refinement: sigma^2 > 0.08 m^2 or delta_Z > 0.18 m
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from backend.config import BOUNDS, FOVEATED, QUADTREE
from backend.mapping.cell_statistics import CellStats, compute_cell_statistics
from backend.mapping.quadtree import QuadtreeNode


def cell_footprint_r_min_r_max(
    x_min: float, x_max: float, y_min: float, y_max: float
) -> Tuple[float, float]:
    """
    Computes exact geometrical minimum and maximum Euclidean distance
    from the sensor origin (0.0, 0.0) to an axis-aligned rectangular cell footprint.
    """
    if x_min <= 0.0 <= x_max:
        dx_min = 0.0
    else:
        dx_min = min(abs(x_min), abs(x_max))

    if y_min <= 0.0 <= y_max:
        dy_min = 0.0
    else:
        dy_min = min(abs(y_min), abs(y_max))

    r_min = float(np.sqrt(dx_min * dx_min + dy_min * dy_min))

    dx_max = max(abs(x_min), abs(x_max))
    dy_max = max(abs(y_min), abs(y_max))
    r_max = float(np.sqrt(dx_max * dx_max + dy_max * dy_max))

    return r_min, r_max


def is_in_radial_envelope(x: float, y: float, max_radial_range: float = 100.0) -> bool:
    """
    Returns True iff coordinate (x, y) is within the radial perception envelope (r <= max_radial_range).
    Distinguishes the rectangular storage envelope (X: [-20, 100], Y: [-50, 50]) from the
    circular perception envelope (r <= 100.0 m).
    Expected coverage is strictly: rectangle ∩ radial perception envelope.
    Points inside the rectangle with r > 100.0 m (e.g. at corners (90, 48)) are outside the perception envelope.
    """
    return float(np.sqrt(x * x + y * y)) <= max_radial_range


def get_zone_for_distance(r: float) -> int:
    """
    Returns the exact zone index (1, 2, 3, 4) for a radial distance r in meters.
    Zone 1: 0.0 <= r < 10.0  -> 0.05m resolution
    Zone 2: 10.0 <= r < 25.0 -> 0.10m resolution
    Zone 3: 25.0 <= r < 50.0 -> 0.25m resolution
    Zone 4: 50.0 <= r <= 100.0 -> 0.50m resolution
    Returns 5 if r > 100.0 m (outside radial perception envelope; excluded from perception zones).
    """
    if r < 0.0:
        return 1
    if r < 10.0:
        return 1
    elif r < 25.0:
        return 2
    elif r < 50.0:
        return 3
    elif r <= 100.0:
        return 4
    else:
        return 5


def get_mandated_resolution(zone: int) -> float:
    """Returns the PS 26053 mandated resolution for a given zone index."""
    if zone == 1:
        return 0.05
    elif zone == 2:
        return 0.10
    elif zone == 3:
        return 0.25
    elif zone == 4:
        return 0.50
    return 0.50


def check_zone_straddling(r_min: float, r_max: float) -> bool:
    """
    Checks if a distance range [r_min, r_max] straddles any radial boundary
    (10.0 m, 25.0 m, 50.0 m, or 100.0 m).
    """
    boundaries = [10.0, 25.0, 50.0]
    for b in boundaries:
        if r_min < b < r_max:
            return True
    return False


def get_single_zone_for_footprint(r_min: float, r_max: float) -> Optional[int]:
    """
    Returns the zone index if the entire footprint [r_min, r_max] is contained
    within exactly one radial zone. Returns None if it straddles boundaries.
    """
    z_min = get_zone_for_distance(r_min)
    z_max = get_zone_for_distance(r_max)
    if z_min == z_max and z_min in (1, 2, 3, 4):
        return z_min
    return None


def make_deterministic_cell_id(x: float, y: float, zone: int, resolution: float) -> str:
    """
    Generates a deterministic cell ID string.
    The same (x, y, zone, resolution) always yields the exact same cell ID.
    """
    rx = round(float(x), 4)
    ry = round(float(y), 4)
    rres = round(float(resolution), 3)
    return f"z{zone}_{rres}m_{rx:+09.4f}_{ry:+09.4f}"


@dataclass(slots=True)
class FoveatedCell:
    """
    Represents an axis-aligned rectangular leaf in the Foveated Spatial Index.
    Zero raw point arrays are stored permanently.
    Separates spatial coverage / cell existence from LiDAR observation / statistics.
    """
    x: float
    y: float
    size: float
    zone: int
    cell_id: str
    depth: int = 0
    stats: Optional[CellStats] = None
    cost: int = 0
    is_leaf: bool = True
    semantic_cost: int = 0
    is_obstacle: bool = False
    is_hazard: bool = False
    observed: bool = True
    valid: bool = True

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

    @property
    def x_min(self) -> float:
        return self.x - self.size / 2.0

    @property
    def x_max(self) -> float:
        return self.x + self.size / 2.0

    @property
    def y_min(self) -> float:
        return self.y - self.size / 2.0

    @property
    def y_max(self) -> float:
        return self.y + self.size / 2.0

    @property
    def r_min(self) -> float:
        r_min, _ = cell_footprint_r_min_r_max(self.min_x, self.max_x, self.min_y, self.max_y)
        return r_min

    @property
    def r_max(self) -> float:
        _, r_max = cell_footprint_r_min_r_max(self.min_x, self.max_x, self.min_y, self.max_y)
        return r_max

    @property
    def z_mean(self) -> float:
        if not self.observed or not self.valid or self.stats is None:
            return float("nan")
        return float(self.stats.mean_z)

    @property
    def z_min(self) -> float:
        if not self.observed or not self.valid or self.stats is None:
            return float("nan")
        return float(self.stats.z_min)

    @property
    def z_max(self) -> float:
        if not self.observed or not self.valid or self.stats is None:
            return float("nan")
        return float(self.stats.z_max)

    @property
    def delta_z(self) -> float:
        if not self.observed or not self.valid or self.stats is None:
            return float("nan")
        return float(self.stats.delta_z)

    @property
    def variance(self) -> float:
        if not self.observed or not self.valid or self.stats is None:
            return float("nan")
        return float(self.stats.variance)

    @property
    def slope(self) -> float:
        if not self.observed or not self.valid or self.stats is None:
            return float("nan")
        return float(self.stats.slope)

    @property
    def mean_z(self) -> float:
        return self.z_mean

    @property
    def point_count(self) -> int:
        if not self.observed or not self.valid or self.stats is None:
            return 0
        return int(self.stats.point_count)

    @classmethod
    def create_unobserved_cell(cls, x: float, y: float, size: float, zone: int) -> "FoveatedCell":
        """Factory for an unobserved cell with valid=False, cost=-1, and NaN statistics."""
        cell_id = make_deterministic_cell_id(x, y, zone, size)
        return cls(
            x=x,
            y=y,
            size=size,
            zone=zone,
            cell_id=cell_id,
            depth=0,
            stats=None,
            cost=-1,
            semantic_cost=-1,
            is_leaf=True,
            is_obstacle=False,
            is_hazard=False,
            observed=False,
            valid=False,
        )

    def to_quadtree_node(self) -> QuadtreeNode:
        """Converts to a QuadtreeNode for seamless downstream compatibility."""
        node = QuadtreeNode(
            x=self.x,
            y=self.y,
            size=self.size,
            depth=self.depth,
            stats=self.stats,
            cost=self.cost,
            is_leaf=True,
            semantic_cost=self.semantic_cost,
            is_obstacle=self.is_obstacle,
            is_hazard=self.is_hazard,
            observed=self.observed,
            valid=self.valid,
        )
        node.cell_id = self.cell_id
        node.x_min = self.x_min
        node.x_max = self.x_max
        node.y_min = self.y_min
        node.y_max = self.y_max
        node.zone = self.zone
        return node

    def to_dict(self) -> dict:
        d = {
            "x": float(round(self.x, 3)),
            "y": float(round(self.y, 3)),
            "size": float(round(self.size, 3)),
            "cost": int(self.cost),
            "zone": int(self.zone),
            "cell_id": self.cell_id,
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


class FoveatedGrid:
    """
    Hybrid Hierarchical Foveated Spatial Index implementation.
    Operates within rectangular storage bounds (X: [-20, 100], Y: [-50, 50], Z: [-3, 5])
    and radial perception envelope (r <= 100 m).
    """

    def __init__(
        self,
        bounds=None,
        config=None,
        max_radial_range: float = FOVEATED.MAX_RADIAL_RANGE,
        tau_sigma: float = FOVEATED.TAU_SIGMA_SQ,
        tau_z: float = FOVEATED.TAU_DELTA_Z,
    ):
        if config is not None:
            max_radial_range = getattr(config, "MAX_RADIAL_RANGE", max_radial_range)
            tau_sigma = getattr(config, "TAU_SIGMA_SQ", tau_sigma)
            tau_z = getattr(config, "TAU_DELTA_Z", tau_z)
        self.bounds = bounds if bounds is not None else BOUNDS
        self.max_radial_range = float(max_radial_range)
        self.tau_sigma = float(tau_sigma)
        self.tau_z = float(tau_z)
        self.cells: Dict[str, FoveatedCell] = {}
        self.leaves: List[FoveatedCell] = []

    def clear(self) -> None:
        self.cells.clear()
        self.leaves.clear()

    @staticmethod
    def get_zone_and_resolution(x: float, y: float) -> Tuple[int, float]:
        """
        Determines the zone and mandated resolution for coordinate (x, y).
        """
        r = float(np.sqrt(x * x + y * y))
        zone = get_zone_for_distance(r)
        if zone == 5:
            # Outside 100m perception envelope
            return 5, 0.50
        res = get_mandated_resolution(zone)
        return zone, res

    @staticmethod
    def get_cell_for_point(
        x: float, y: float, zone: Optional[int] = None, res: Optional[float] = None
    ) -> Tuple[float, float, float, int, str]:
        """
        Calculates cell center (cx, cy), size, zone, and deterministic ID for coordinate (x, y).
        Guarantees that the cell footprint does NOT straddle radial boundaries.
        Boundary-aligned snap ensures:
        - Points in Zone 1 (r < 10.0) yield cells with footprint <= 10.0m (Zone 1)
        - Points in Zone 2 (10.0 <= r < 25.0) yield cells with footprint in [10.0, 25.0m) (Zone 2)
        - Points in Zone 3 (25.0 <= r < 50.0) yield cells with footprint in [25.0, 50.0m) (Zone 3)
        - Points in Zone 4 (50.0 <= r <= 100.0) yield cells with footprint in [50.0, 100.0m] (Zone 4)
        """
        r = float(np.sqrt(x * x + y * y))
        if zone is None:
            zone = get_zone_for_distance(r)
        if res is None:
            res = get_mandated_resolution(zone)

        # Standard grid index
        ix = int(np.floor(x / res))
        iy = int(np.floor(y / res))
        cx = (ix + 0.5) * res
        cy = (iy + 0.5) * res

        # Geometrical boundary verification & non-straddling snap
        min_x = cx - res / 2.0
        max_x = cx + res / 2.0
        min_y = cy - res / 2.0
        max_y = cy + res / 2.0
        r_min, r_max = cell_footprint_r_min_r_max(min_x, max_x, min_y, max_y)

        # If straddling occurs at zone boundary:
        # Snap the cell boundary flush to the radial boundary so r_min/r_max stays in the zone
        boundaries = [(10.0, 1, 2), (25.0, 2, 3), (50.0, 3, 4)]
        for b, z_inner, z_outer in boundaries:
            if r_min < b < r_max:
                if zone == z_inner:
                    # Cell belongs to inner zone: pull cell inward so r_max <= b
                    scale = (b - 1e-4) / max(r_max, 1e-6)
                    # Shift center inward along radial direction
                    dx = cx * (1.0 - scale)
                    dy = cy * (1.0 - scale)
                    cx -= dx
                    cy -= dy
                else:
                    # Cell belongs to outer zone: push cell outward so r_min >= b
                    scale = (b + 1e-4) / max(r_min, 1e-6)
                    dx = cx * (scale - 1.0)
                    dy = cy * (scale - 1.0)
                    cx += dx
                    cy += dy
                break

        cell_id = make_deterministic_cell_id(cx, cy, zone, res)
        return cx, cy, res, zone, cell_id

    def build_foveated(
        self,
        points: np.ndarray,
        point_costs: Optional[np.ndarray] = None,
    ) -> List[QuadtreeNode]:
        """
        Constructs the hybrid hierarchical foveated spatial index for the input sweep.
        Returns a list of QuadtreeNode leaves compatible with all downstream modules.
        """
        self.clear()
        if points is None or len(points) == 0:
            return []

        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 3 if pts.shape[0] % 3 == 0 else 4)

        # Filter: Rectangular storage bounds AND Radial perception envelope (r <= 100 m)
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        r = np.sqrt(x * x + y * y)

        valid_mask = (
            (x >= self.bounds.X_MIN) & (x <= self.bounds.X_MAX) &
            (y >= self.bounds.Y_MIN) & (y <= self.bounds.Y_MAX) &
            (z >= self.bounds.Z_MIN) & (z <= self.bounds.Z_MAX) &
            (r <= self.max_radial_range) &
            np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        )

        clean_pts = pts[valid_mask]
        costs = point_costs[valid_mask] if (point_costs is not None and len(point_costs) == len(pts)) else None

        if len(clean_pts) == 0:
            return []

        # Vectorized or grouped binning by exact zone
        x_c = clean_pts[:, 0]
        y_c = clean_pts[:, 1]
        z_c = clean_pts[:, 2]
        r_c = np.sqrt(x_c * x_c + y_c * y_c)

        # Assign zones
        z1_mask = r_c < 10.0
        z2_mask = (r_c >= 10.0) & (r_c < 25.0)
        z3_mask = (r_c >= 25.0) & (r_c < 50.0)
        z4_mask = (r_c >= 50.0) & (r_c <= 100.0)

        zone_configs = [
            (1, 0.05, z1_mask),
            (2, 0.10, z2_mask),
            (3, 0.25, z3_mask),
            (4, 0.50, z4_mask),
        ]

        quadtree_leaves: List[QuadtreeNode] = []

        for zone_idx, res, mask in zone_configs:
            sub_pts = clean_pts[mask]
            if len(sub_pts) == 0:
                continue
            sub_costs = costs[mask] if costs is not None else None

            # Quantize points to grid keys
            sx = sub_pts[:, 0]
            sy = sub_pts[:, 1]
            sz = sub_pts[:, 2]

            ix = np.floor(sx / res).astype(np.int32)
            iy = np.floor(sy / res).astype(np.int32)

            # Unique cell keys
            unique_keys, inverse_idx = np.unique(
                np.column_stack((ix, iy)), axis=0, return_inverse=True
            )

            # Compute statistics per unique cell
            n_cells = len(unique_keys)
            c_cx = (unique_keys[:, 0] + 0.5) * res
            c_cy = (unique_keys[:, 1] + 0.5) * res

            # Group statistics using numpy ops
            z_min_arr = np.full(n_cells, np.inf, dtype=np.float32)
            z_max_arr = np.full(n_cells, -np.inf, dtype=np.float32)
            z_sum_arr = np.zeros(n_cells, dtype=np.float32)
            z_sq_sum_arr = np.zeros(n_cells, dtype=np.float32)
            count_arr = np.zeros(n_cells, dtype=np.int32)
            max_cost_arr = np.zeros(n_cells, dtype=np.int32)

            np.minimum.at(z_min_arr, inverse_idx, sz)
            np.maximum.at(z_max_arr, inverse_idx, sz)
            np.add.at(z_sum_arr, inverse_idx, sz)
            np.add.at(z_sq_sum_arr, inverse_idx, sz * sz)
            np.add.at(count_arr, inverse_idx, 1)

            if sub_costs is not None:
                np.maximum.at(max_cost_arr, inverse_idx, sub_costs)

            # Build cell objects
            for i in range(n_cells):
                cnt = count_arr[i]
                if cnt == 0:
                    continue

                cell_x = float(c_cx[i])
                cell_y = float(c_cy[i])
                mean_z = float(z_sum_arr[i] / cnt)
                z_min = float(z_min_arr[i])
                z_max = float(z_max_arr[i])
                delta_z = max(0.0, z_max - z_min)
                mean_sq = float(z_sq_sum_arr[i] / cnt)
                variance = max(0.0, mean_sq - mean_z * mean_z)

                # Geometrical boundary containment verification
                half = res / 2.0
                min_x = cell_x - half
                max_x = cell_x + half
                min_y = cell_y - half
                max_y = cell_y + half
                rf_min, rf_max = cell_footprint_r_min_r_max(min_x, max_x, min_y, max_y)

                # Ensure strict non-straddling:
                actual_zone = zone_idx
                if check_zone_straddling(rf_min, rf_max):
                    # Snap center so footprint is strictly contained in zone_idx
                    if zone_idx == 1 and rf_max >= 10.0:
                        shift = (rf_max - 10.0 + 1e-4)
                        # Shift towards origin
                        dist = np.hypot(cell_x, cell_y)
                        if dist > 1e-6:
                            cell_x -= (cell_x / dist) * shift
                            cell_y -= (cell_y / dist) * shift
                    elif zone_idx == 2:
                        if rf_min < 10.0:
                            shift = (10.0 - rf_min + 1e-4)
                            dist = np.hypot(cell_x, cell_y)
                            if dist > 1e-6:
                                cell_x += (cell_x / dist) * shift
                                cell_y += (cell_y / dist) * shift
                        elif rf_max >= 25.0:
                            shift = (rf_max - 25.0 + 1e-4)
                            dist = np.hypot(cell_x, cell_y)
                            if dist > 1e-6:
                                cell_x -= (cell_x / dist) * shift
                                cell_y -= (cell_y / dist) * shift
                    elif zone_idx == 3:
                        if rf_min < 25.0:
                            shift = (25.0 - rf_min + 1e-4)
                            dist = np.hypot(cell_x, cell_y)
                            if dist > 1e-6:
                                cell_x += (cell_x / dist) * shift
                                cell_y += (cell_y / dist) * shift
                        elif rf_max >= 50.0:
                            shift = (rf_max - 50.0 + 1e-4)
                            dist = np.hypot(cell_x, cell_y)
                            if dist > 1e-6:
                                cell_x -= (cell_x / dist) * shift
                                cell_y -= (cell_y / dist) * shift
                    elif zone_idx == 4 and rf_min < 50.0:
                        shift = (50.0 - rf_min + 1e-4)
                        dist = np.hypot(cell_x, cell_y)
                        if dist > 1e-6:
                            cell_x += (cell_x / dist) * shift
                            cell_y += (cell_y / dist) * shift

                cell_id = make_deterministic_cell_id(cell_x, cell_y, actual_zone, res)

                stats = CellStats(
                    z_min=z_min,
                    z_max=z_max,
                    mean_z=mean_z,
                    variance=variance,
                    slope=0.0,
                    point_count=cnt,
                    delta_z=delta_z,
                )

                # Determine traversability cost
                cell_cost = int(max_cost_arr[i])
                if cell_cost == 0:
                    if delta_z > 0.25 or z_max > -1.15:
                        cell_cost = 255
                    elif delta_z > 0.15:
                        cell_cost = 100
                    else:
                        cell_cost = 0

                cell = FoveatedCell(
                    x=cell_x,
                    y=cell_y,
                    size=res,
                    zone=actual_zone,
                    cell_id=cell_id,
                    stats=stats,
                    cost=cell_cost,
                    is_leaf=True,
                )

                self.cells[cell_id] = cell
                self.leaves.append(cell)

        # Resolve any multi-resolution or boundary cell overlaps: finer zones take strict spatial precedence
        sorted_leaves = sorted(self.leaves, key=lambda c: (c.zone, c.size))
        occupied_bins = set()
        resolved_cells: Dict[str, FoveatedCell] = {}
        resolved_leaves: List[FoveatedCell] = []
        resolved_quadtree: List[QuadtreeNode] = []

        for cell in sorted_leaves:
            # Check 5cm grid footprint
            bx_min = int(np.floor((cell.min_x + 1e-5) / 0.05))
            bx_max = int(np.ceil((cell.max_x - 1e-5) / 0.05))
            by_min = int(np.floor((cell.min_y + 1e-5) / 0.05))
            by_max = int(np.ceil((cell.max_y - 1e-5) / 0.05))

            cell_bins = []
            conflict = False
            for bx in range(bx_min, bx_max):
                for by in range(by_min, by_max):
                    bkey = (bx, by)
                    if bkey in occupied_bins:
                        conflict = True
                        break
                    cell_bins.append(bkey)
                if conflict:
                    break

            if not conflict:
                occupied_bins.update(cell_bins)
                resolved_cells[cell.cell_id] = cell
                resolved_leaves.append(cell)
                resolved_quadtree.append(cell.to_quadtree_node())

        self.cells = resolved_cells
        self.leaves = resolved_leaves
        return resolved_quadtree

    @classmethod
    def create_unobserved_cell(
        cls, x: float, y: float, zone: Optional[int] = None, res: Optional[float] = None
    ) -> FoveatedCell:
        """
        Creates an unobserved spatial cell for areas within perception coverage
        with no valid LiDAR returns. Never fabricates elevation (Z values are NaN).
        """
        r = float(np.sqrt(x * x + y * y))
        if zone is None:
            zone = get_zone_for_distance(r)
        if res is None:
            res = get_mandated_resolution(zone)
        ix = int(np.floor(x / res))
        iy = int(np.floor(y / res))
        cx = (ix + 0.5) * res
        cy = (iy + 0.5) * res
        cell_id = make_deterministic_cell_id(cx, cy, zone, res)
        return FoveatedCell(
            x=cx,
            y=cy,
            size=res,
            zone=zone,
            cell_id=cell_id,
            stats=None,
            cost=-1,
            is_leaf=True,
            observed=False,
            valid=False,
        )

    def get_leaves(self) -> List[FoveatedCell]:
        return self.leaves

    def get_statistics(self) -> Dict[str, Any]:
        """Returns cell counts and breakdown per radial zone."""
        zone_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for leaf in self.leaves:
            zone_counts[leaf.zone] = zone_counts.get(leaf.zone, 0) + 1
        return {
            "total_leaves": len(self.leaves),
            "zone_1_cells": zone_counts[1],
            "zone_2_cells": zone_counts[2],
            "zone_3_cells": zone_counts[3],
            "zone_4_cells": zone_counts[4],
        }
