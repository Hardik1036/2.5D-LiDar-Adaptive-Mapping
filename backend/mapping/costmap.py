"""
Traversability Costmap Evaluator.
Translates 2.5D cell elevation metrics (slope, delta_z, roughness)
and dynamic obstacle presences into traversability cost values [0, 255].
  - 0 to 50: Safe traversable terrain
  - 51 to 180: Caution / steep slope / rough ground
  - 181 to 255: Lethal obstacle / non-traversable
"""

from typing import List, Optional
import numpy as np

from backend.config import COSTMAP
from backend.mapping.quadtree import QuadtreeNode


class CostmapEvaluator:
    """
    Evaluates terrain traversability costs for 2.5D quadtree leaf cells.
    """

    def __init__(
        self,
        safe_max: int = COSTMAP.SAFE_MAX,
        caution_max: int = COSTMAP.CAUTION_MAX,
        lethal_val: int = COSTMAP.LETHAL_VAL,
        max_traversable_slope: float = COSTMAP.MAX_TRAVERSABLE_SLOPE_DEG,
        lethal_slope: float = COSTMAP.LETHAL_SLOPE_DEG,
        max_step_height: float = COSTMAP.MAX_STEP_HEIGHT,
        lethal_step_height: float = COSTMAP.LETHAL_STEP_HEIGHT,
    ):
        self.safe_max = safe_max
        self.caution_max = caution_max
        self.lethal_val = lethal_val
        self.max_slope = max_traversable_slope
        self.lethal_slope = lethal_slope
        self.max_step = max_step_height
        self.lethal_step = lethal_step_height

    def evaluate_node_cost(self, node: QuadtreeNode) -> int:
        """
        Computes cost in [0, 255] for a single quadtree leaf node based on its CellStats.
        """
        if node.stats is None:
            return 0

        stats = node.stats
        slope = stats.slope
        delta_z = stats.delta_z

        # 1. Step height evaluation
        if delta_z >= self.lethal_step:
            return self.lethal_val

        step_ratio = max(0.0, (delta_z - self.max_step) / (self.lethal_step - self.max_step)) if self.lethal_step > self.max_step else 1.0

        # 2. Slope evaluation
        if slope >= self.lethal_slope:
            return self.lethal_val
        elif slope <= self.max_slope:
            slope_cost = (slope / self.max_slope) * self.safe_max
        else:
            slope_ratio = (slope - self.max_slope) / (self.lethal_slope - self.max_slope)
            slope_cost = self.safe_max + slope_ratio * (self.caution_max - self.safe_max)

        combined = max(slope_cost, step_ratio * self.caution_max)
        val = int(combined)
        return 255 if val > 255 else (0 if val < 0 else val)

    def evaluate_leaves(self, leaves: List[QuadtreeNode]):
        """Evaluates cost for all leaf nodes in place."""
        for leaf in leaves:
            leaf.cost = self.evaluate_node_cost(leaf)

    def apply_obstacle_occupancy(self, leaves: List[QuadtreeNode], obstacle_points: np.ndarray):
        """
        Flags quadtree cells containing non-ground obstacle points as lethal (255).
        High-speed spatial hashing in < 1ms.
        """
        if len(leaves) == 0 or len(obstacle_points) == 0:
            return

        from backend.config import BOUNDS, QUADTREE

        ox = obstacle_points[:, 0]
        oy = obstacle_points[:, 1]

        num_x = int(np.ceil((BOUNDS.X_MAX - BOUNDS.X_MIN) / QUADTREE.COARSE_RESOLUTION))
        num_y = int(np.ceil((BOUNDS.Y_MAX - BOUNDS.Y_MIN) / QUADTREE.COARSE_RESOLUTION))

        o_ix = np.clip(((ox - BOUNDS.X_MIN) / QUADTREE.COARSE_RESOLUTION).astype(np.int32), 0, num_x - 1)
        o_iy = np.clip(((oy - BOUNDS.Y_MIN) / QUADTREE.COARSE_RESOLUTION).astype(np.int32), 0, num_y - 1)
        obs_set = set((o_iy * num_x + o_ix).tolist())

        res = QUADTREE.COARSE_RESOLUTION
        x_min = BOUNDS.X_MIN
        y_min = BOUNDS.Y_MIN

        for leaf in leaves:
            l_ix = int((leaf.x - x_min) / res)
            l_iy = int((leaf.y - y_min) / res)
            if 0 <= l_ix < num_x and 0 <= l_iy < num_y:
                if (l_iy * num_x + l_ix) in obs_set:
                    leaf.cost = self.lethal_val

    def apply_dynamic_hazards(self, leaves: List[QuadtreeNode], hazard_circles_or_ellipses: List[dict]):
        """
        Inflates cell costs within predicted dynamic obstacle hazard areas.
        Each hazard item contains {x, y, radius, cost}.
        Accelerated with spatial hashing in < 1.5 ms.
        """
        if not leaves or not hazard_circles_or_ellipses:
            return

        res = 0.50  # Spatial coarse bin size for fast O(1) leaf candidate indexing
        inv_res = 1.0 / res
        grid = {}
        for leaf in leaves:
            k = (int(round(leaf.x * inv_res)), int(round(leaf.y * inv_res)))
            grid.setdefault(k, []).append(leaf)

        for hazard in hazard_circles_or_ellipses:
            hx = hazard["x"]
            hy = hazard["y"]
            r = hazard["radius"]
            hz_cost = hazard.get("cost", self.lethal_val)
            r2 = r * r

            min_gx = int(round((hx - r) * inv_res))
            max_gx = int(round((hx + r) * inv_res))
            min_gy = int(round((hy - r) * inv_res))
            max_gy = int(round((hy + r) * inv_res))

            for gx in range(min_gx, max_gx + 1):
                for gy in range(min_gy, max_gy + 1):
                    cell_leaves = grid.get((gx, gy))
                    if cell_leaves:
                        for leaf in cell_leaves:
                            dx = leaf.x - hx
                            dy = leaf.y - hy
                            if (dx * dx + dy * dy) <= r2:
                                if leaf.cost < hz_cost:
                                    leaf.cost = hz_cost

