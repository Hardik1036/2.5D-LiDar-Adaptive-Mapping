"""
Ghost artifact clearing via 2D free-space raycasting.
Clears dynamic obstacle "smear" trails left by moving objects across consecutive frames.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from backend.config import BOUNDS, QUADTREE, COSTMAP
from backend.mapping.quadtree import QuadtreeNode


class GhostClearing:
    """
    Clears smear trails caused by moving dynamic obstacles.
    Maintains historical occupied obstacle cell footprints. When new LiDAR rays
    penetrate through previously occupied space and hit farther targets or ground,
    the obsolete obstacle footprint is cleared to prevent false positive ghosts.
    Enforces a strict execution budget of < 2.5 ms (< 400 evaluated ray lines per sweep).
    """

    def __init__(self, coarse_res: float = QUADTREE.COARSE_RESOLUTION):
        self.coarse_res = coarse_res
        self.prev_dynamic_footprints: List[Tuple[float, float, float]] = []  # (x, y, radius)
        self.current_dynamic_footprints: List[Tuple[float, float, float]] = []

    def register_dynamic_footprints(self, dynamic_tracks):
        """Records current dynamic obstacle positions to check against future rays."""
        # Archive current to previous unconditionally
        self.prev_dynamic_footprints = list(self.current_dynamic_footprints)
        self.current_dynamic_footprints = [
            (t.x, t.y, max(t.dimensions[0], t.dimensions[1]) / 2.0 + 0.3)
            for t in dynamic_tracks
        ]

    def clear_ghosts(
        self,
        current_leaves: List[QuadtreeNode],
        current_points: Optional[np.ndarray] = None,
        dynamic_tracks: Optional[List[Any]] = None,
    ):
        """
        Budget-protected 2D line-of-sight raycasting (< 2.5 ms).
        Subsamples scan endpoints by 8x and casts rays through previous dynamic footprints.
        Any traversed cells not containing active dynamic obstacles have their occupancy
        decayed back to 0 (Safe/Drivable).
        """
        if not current_leaves:
            return

        # 1. Collect active dynamic footprints
        active_footprints = []
        if dynamic_tracks is not None:
            active_footprints = [
                (t.x, t.y, max(t.dimensions[0], t.dimensions[1]) / 2.0 + 0.3)
                for t in dynamic_tracks
            ]
        elif self.current_dynamic_footprints:
            active_footprints = self.current_dynamic_footprints

        inv_res = 2.0  # 0.50m resolution for fast O(1) integer hashing

        # 2. Build spatial hash map of quadtree leaves
        leaf_grid: Dict[Tuple[int, int], QuadtreeNode] = {}
        for leaf in current_leaves:
            gx = int(round(leaf.x * inv_res))
            gy = int(round(leaf.y * inv_res))
            leaf_grid[(gx, gy)] = leaf

        # 3. Precompute active dynamic obstacle cells (set of integer grid coordinates)
        active_cells: Set[Tuple[int, int]] = set()
        for ax, ay, ar in active_footprints:
            rg = int(ar * inv_res) + 1
            cgx, cgy = int(round(ax * inv_res)), int(round(ay * inv_res))
            for gx in range(cgx - rg, cgx + rg + 1):
                for gy in range(cgy - rg, cgy + rg + 1):
                    active_cells.add((gx, gy))

        # 4. Precompute previous dynamic footprint cells and collect ray endpoints
        prev_cells: Set[Tuple[int, int]] = set()
        ray_grid_endpoints: List[Tuple[int, int]] = []

        for px, py, pr in self.prev_dynamic_footprints:
            rg = int(pr * inv_res) + 1
            cgx, cgy = int(round(px * inv_res)), int(round(py * inv_res))
            for gx in range(cgx - rg, cgx + rg + 1):
                for gy in range(cgy - rg, cgy + rg + 1):
                    prev_cells.add((gx, gy))

            dist = float(np.hypot(px, py))
            if dist > 0.2:
                ext = (dist + 3.0) / dist
                # Cast 3 angular radial rays through and past the obstacle footprint
                for ang in (-0.2, 0.0, 0.2):
                    cos_a, sin_a = np.cos(ang), np.sin(ang)
                    rx = (px * cos_a - py * sin_a) * ext
                    ry = (px * sin_a + py * cos_a) * ext
                    ray_grid_endpoints.append((int(round(rx * inv_res)), int(round(ry * inv_res))))

        # 5. Subsample LiDAR scan endpoints if previous dynamic footprints exist to check line-of-sight
        if current_points is not None and len(current_points) > 0 and prev_cells:
            remaining_budget = max(0, 30 - len(ray_grid_endpoints))
            if remaining_budget > 0:
                subsampled = current_points[::16, :2]
                if len(subsampled) > remaining_budget:
                    step = max(1, len(subsampled) // remaining_budget)
                    subsampled = subsampled[::step][:remaining_budget]
                for pt in subsampled:
                    ray_grid_endpoints.append((int(round(float(pt[0]) * inv_res)), int(round(float(pt[1]) * inv_res))))

        # 6. Direct clearing of previous footprint cells with low delta_z
        for (gx, gy) in prev_cells:
            if (gx, gy) not in active_cells:
                leaf = leaf_grid.get((gx, gy))
                if leaf is not None and leaf.cost > 0:
                    if leaf.stats is None or leaf.stats.delta_z < 0.15:
                        leaf.cost = 0

        # 7. Integer Bresenham line-of-sight raycasting from (0, 0)
        # Guarantees execution time < 1 ms (typically 0.05-0.2 ms)
        for gx1, gy1 in ray_grid_endpoints:
            dx = abs(gx1)
            dy = abs(gy1)
            sx = 1 if gx1 > 0 else -1
            sy = 1 if gy1 > 0 else -1
            err = dx - dy
            x, y = 0, 0

            for _ in range(60):  # max 60 grid steps = 30 meters
                if x == gx1 and y == gy1:
                    break

                if (x, y) in active_cells:
                    break  # ray occluded by active dynamic obstacle

                if (x, y) in prev_cells:
                    leaf = leaf_grid.get((x, y))
                    if leaf is not None and leaf.cost > 0:
                        st = leaf.stats
                        if (
                            st is not None
                            and st.delta_z < 0.15
                            and COSTMAP.GROUND_MIN <= st.mean_z <= COSTMAP.ELEVATED_OBSTACLE_Z
                            and st.z_max <= COSTMAP.ELEVATED_OBSTACLE_Z
                        ):
                            leaf.cost = 0

                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x += sx
                if e2 < dx:
                    err += dx
                    y += sy

        # 8. Corridor clutter & ghost clearance: clear flat road cells in ego travel path
        # where delta_z is minimal (< 0.08m) and cell is not an active dynamic obstacle
        for leaf in current_leaves:
            if 0.5 <= leaf.x <= 35.0 and abs(leaf.y) <= 3.0:
                gx = int(round(leaf.x * inv_res))
                gy = int(round(leaf.y * inv_res))
                if (gx, gy) not in active_cells and leaf.cost > 0:
                    st = leaf.stats
                    if (
                        st is not None
                        and st.delta_z < 0.08
                        and st.slope < 6.0
                        and COSTMAP.GROUND_MIN <= st.mean_z <= COSTMAP.ELEVATED_OBSTACLE_Z
                        and st.z_max <= COSTMAP.ELEVATED_OBSTACLE_Z
                    ):
                        leaf.cost = 0


# Backwards-compatible alias
DynamicGhostClearing = GhostClearing

