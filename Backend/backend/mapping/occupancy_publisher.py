"""
ROS 2 nav_msgs/msg/OccupancyGrid 2D Costmap Publisher / Generator.
Bridges 2.5D Adaptive Quadtree elevation maps into ROS 2 Nav2 standard planar grids
for DRDO SIH 2026 Problem Statement 53.
"""

import time
from typing import Any, Dict, List, Optional
import numpy as np

from backend.config import NAV2, Nav2Config
from backend.mapping.quadtree import QuadtreeNode


class OccupancyGridBuilder:
    """
    Translates variable-resolution 2.5D Quadtree leaf cells into a deterministic
    dense 2D planar OccupancyGrid compliant with ROS 2 `nav_msgs/msg/OccupancyGrid`.
    
    Grid Value Standard (ROS 2 Nav2):
      - 0: Free / traversable terrain
      - 50: Caution / slope / rough terrain
      - 100: Lethal obstacle / non-traversable
      - -1: Unknown / unobserved space
    """

    def __init__(self, config: Nav2Config = NAV2):
        self.config = config
        self.width = config.WIDTH
        self.height = config.HEIGHT
        self.res = config.RESOLUTION
        self.inv_res = 1.0 / self.res
        self.origin_x = config.ORIGIN_X
        self.origin_y = config.ORIGIN_Y
        self.frame_id = config.FRAME_ID

    def build_grid(
        self,
        leaves: List[QuadtreeNode],
        dynamic_hazards: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        """
        Rasterizes quadtree leaves and dynamic obstacle hazard footprints
        into a dense (HEIGHT, WIDTH) int8 occupancy grid.
        Executes in < 3.0 ms for typical 400x400 grid setups.
        """
        # Initialize grid with UNKNOWN (-1)
        grid = np.full((self.height, self.width), self.config.COST_UNKNOWN, dtype=np.int8)

        if not leaves:
            return grid

        inv_r = self.inv_res
        ox = self.origin_x
        oy = self.origin_y
        w_max = self.width - 1
        h_max = self.height - 1

        cost_free = self.config.COST_FREE
        cost_caution = self.config.COST_CAUTION
        cost_lethal = self.config.COST_LETHAL

        # Rasterize quadtree leaves
        for leaf in leaves:
            cost = leaf.cost
            if cost <= 50:
                grid_val = cost_free
            elif cost <= 180:
                grid_val = cost_caution
            else:
                grid_val = cost_lethal

            half_sz = leaf.size * 0.5
            min_c = int((leaf.x - half_sz - ox) * inv_r)
            max_c = int((leaf.x + half_sz - ox) * inv_r)
            min_r = int((leaf.y - half_sz - oy) * inv_r)
            max_r = int((leaf.y + half_sz - oy) * inv_r)

            # Clamp boundaries
            if min_c < 0: min_c = 0
            if max_c > w_max: max_c = w_max
            if min_r < 0: min_r = 0
            if max_r > h_max: max_r = h_max

            if min_c <= max_c and min_r <= max_r:
                grid[min_r:max_r + 1, min_c:max_c + 1] = grid_val

        # Rasterize dynamic obstacle projected hazard cones
        if dynamic_hazards:
            for hz in dynamic_hazards:
                hx = hz.get("x", 0.0)
                hy = hz.get("y", 0.0)
                hr = hz.get("radius", 0.5)
                hz_cost = cost_lethal if hz.get("cost", 255) > 180 else cost_caution

                min_c = max(0, int((hx - hr - ox) * inv_r))
                max_c = min(w_max, int((hx + hr - ox) * inv_r))
                min_r = max(0, int((hy - hr - oy) * inv_r))
                max_r = min(h_max, int((hy + hr - oy) * inv_r))

                if min_c <= max_c and min_r <= max_r:
                    grid[min_r:max_r + 1, min_c:max_c + 1] = hz_cost

        return grid

    def to_ros_message_dict(
        self,
        grid: np.ndarray,
        timestamp: Optional[float] = None,
        flatten_data: bool = True,
    ) -> Dict[str, Any]:
        """
        Formats grid into a standard dictionary schema matching `nav_msgs/msg/OccupancyGrid`.
        Can be serialized directly to JSON or ingested by ROS 2 Python nodes (rclpy).
        """
        ts = timestamp if timestamp is not None else time.time()
        sec = int(ts)
        nanosec = int((ts - sec) * 1e9)

        data_payload = grid.flatten().tolist() if flatten_data else grid

        return {
            "header": {
                "stamp": {"sec": sec, "nanosec": nanosec},
                "frame_id": self.frame_id,
            },
            "info": {
                "map_load_time": {"sec": sec, "nanosec": nanosec},
                "resolution": self.res,
                "width": self.width,
                "height": self.height,
                "origin": {
                    "position": {
                        "x": self.origin_x,
                        "y": self.origin_y,
                        "z": 0.0,
                    },
                    "orientation": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0,
                        "w": 1.0,
                    },
                },
            },
            "data": data_payload,
        }
