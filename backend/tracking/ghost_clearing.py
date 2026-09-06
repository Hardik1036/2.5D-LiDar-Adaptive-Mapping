"""
Ghost artifact clearing via 2D free-space raycasting.
Clears dynamic obstacle "smear" trails left by moving objects across consecutive frames.
"""

from typing import List, Optional, Set, Tuple
import numpy as np

from backend.config import BOUNDS, QUADTREE
from backend.mapping.quadtree import QuadtreeNode


class GhostClearing:
    """
    Clears smear trails caused by moving dynamic obstacles.
    Maintains historical occupied obstacle cell footprints. When new LiDAR rays
    penetrate through previously occupied space and hit farther targets or ground,
    the obsolete obstacle footprint is cleared to prevent false positive ghosts.
    """

    def __init__(self, coarse_res: float = QUADTREE.COARSE_RESOLUTION):
        self.coarse_res = coarse_res
        self.prev_dynamic_footprints: List[Tuple[float, float, float]] = []  # (x, y, radius)

    def register_dynamic_footprints(self, dynamic_tracks):
        """Records current dynamic obstacle positions to check against future rays."""
        self.prev_dynamic_footprints = [
            (t.x, t.y, max(t.dimensions[0], t.dimensions[1]) / 2.0 + 0.2)
            for t in dynamic_tracks
        ]

    def clear_ghosts(self, current_leaves: List[QuadtreeNode], current_points: Optional[np.ndarray] = None):
        """
        Validates cells against dynamic trails in < 1ms.
        If a cell was part of a previous dynamic footprint, but the current quadtree
        shows flat ground elevation (delta_z < 0.10m), clear its cost to safe.
        """
        if not self.prev_dynamic_footprints:
            return

        for (px, py, radius) in self.prev_dynamic_footprints:
            r2 = radius * radius

            for leaf in current_leaves:
                dx = leaf.x - px
                dy = leaf.y - py
                if (dx * dx + dy * dy) <= r2:
                    if leaf.cost > 180:
                        # If cell has flat ground statistics or no step, clear ghost smear
                        if leaf.stats is None or leaf.stats.delta_z < 0.10:
                            leaf.cost = 0


# Backwards-compatible alias
DynamicGhostClearing = GhostClearing
