"""
Sensor Confidence & Degraded-Mode Safe Fallback (Feature F5.3).
Monitors per-quadrant LiDAR return density against historical baselines to detect
lens occlusions, thick dust/smoke blindness, or sensor hardware degradation.
Triggers defensive costmap inflation (cost = 180) and preserves terrain memory via dead-reckoning.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from backend.mapping.quadtree import QuadtreeNode
from backend.mapping.temporal_blender import TemporalMapBlender


class SensorHealthMonitor:
    """
    Evaluates field-of-view (FOV) return density across angular quadrants.
    Detects severe sensor occlusion (smoke, mud, dust blindness) when return counts
    drop below 15% of historical moving averages.
    """

    STATUS_HEALTHY = "HEALTHY"
    STATUS_DEGRADED = "DEGRADED"
    ALERT_CAUTION_ACTIVE = "DEGRADED_VISIBILITY_CAUTION_ACTIVE"
    ALERT_NOMINAL = "ALL_SYSTEMS_NOMINAL"

    # 4 FOV Quadrants: Front-Right, Front-Left, Rear-Left, Rear-Right
    QUADRANT_NAMES = ["FRONT_RIGHT", "FRONT_LEFT", "REAR_LEFT", "REAR_RIGHT"]
    FORWARD_QUADRANTS = [0, 1]  # Front-Right, Front-Left

    def __init__(
        self,
        alpha: float = 0.08,             # Moving average smoothing factor
        min_baseline_pts: int = 80,      # Minimum points required to establish valid baseline
        drop_threshold_ratio: float = 0.15,  # Alert triggered if count < 15% of moving average
        degraded_caution_cost: int = 180,    # Inflated caution cost for blind zones
    ):
        self.alpha = alpha
        self.min_baseline_pts = min_baseline_pts
        self.drop_threshold = drop_threshold_ratio
        self.caution_cost = degraded_caution_cost

        # Moving average of points per quadrant: [FR, FL, RL, RR]
        self._historical_averages = np.zeros(4, dtype=np.float32)
        self._initialized = False

        self.last_status = self.STATUS_HEALTHY
        self.last_system_alert = self.ALERT_NOMINAL
        self.last_degraded_quadrants: List[int] = []

    def update(self, points: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Updates FOV quadrant density counters and determines sensor operational health.

        Args:
            points: (N, 3) or (N, 4) incoming point cloud array.

        Returns:
            Dictionary containing health status, alert strings, and per-quadrant metrics.
        """
        quadrant_counts = np.zeros(4, dtype=np.int32)

        if points is not None and len(points) > 0:
            px = points[:, 0]
            py = points[:, 1]
            angles = np.arctan2(py, px)  # [-pi, pi]

            # Vectorized quadrant assignment:
            # Q0 (Front-Right): [0, pi/2]
            # Q1 (Front-Left) : [pi/2, pi]
            # Q2 (Rear-Left)  : [-pi, -pi/2]
            # Q3 (Rear-Right) : [-pi/2, 0]
            m0 = (angles >= 0.0) & (angles < np.pi * 0.5)
            m1 = (angles >= np.pi * 0.5) & (angles <= np.pi)
            m2 = (angles >= -np.pi) & (angles < -np.pi * 0.5)
            m3 = (angles >= -np.pi * 0.5) & (angles < 0.0)

            quadrant_counts[0] = int(np.count_nonzero(m0))
            quadrant_counts[1] = int(np.count_nonzero(m1))
            quadrant_counts[2] = int(np.count_nonzero(m2))
            quadrant_counts[3] = int(np.count_nonzero(m3))

        # Initialize or update historical moving averages
        if not self._initialized:
            if np.sum(quadrant_counts) > self.min_baseline_pts:
                self._historical_averages = quadrant_counts.astype(np.float32).copy()
                self._initialized = True
        else:
            for q in range(4):
                # Update moving average with exponential smoothing
                self._historical_averages[q] = (
                    (1.0 - self.alpha) * self._historical_averages[q]
                    + self.alpha * float(quadrant_counts[q])
                )

        # Check for degradation in forward quadrants (FR, FL)
        degraded_quadrants = []
        if self._initialized:
            for q in self.FORWARD_QUADRANTS:
                baseline = self._historical_averages[q]
                count = quadrant_counts[q]
                if baseline >= self.min_baseline_pts:
                    if count < (self.drop_threshold * baseline):
                        degraded_quadrants.append(q)

        is_degraded = len(degraded_quadrants) > 0
        self.last_status = self.STATUS_DEGRADED if is_degraded else self.STATUS_HEALTHY
        self.last_system_alert = self.ALERT_CAUTION_ACTIVE if is_degraded else self.ALERT_NOMINAL
        self.last_degraded_quadrants = degraded_quadrants

        return {
            "status": self.last_status,
            "system_status": self.last_system_alert,
            "is_degraded": is_degraded,
            "degraded_quadrants": [self.QUADRANT_NAMES[q] for q in degraded_quadrants],
            "quadrant_counts": {self.QUADRANT_NAMES[i]: int(quadrant_counts[i]) for i in range(4)},
            "historical_averages": {self.QUADRANT_NAMES[i]: float(round(self._historical_averages[i], 1)) for i in range(4)},
        }

    def apply_degraded_costmap_inflation(
        self,
        leaves: List[QuadtreeNode],
        degraded_quadrants: Optional[List[int]] = None,
        caution_cost: Optional[int] = None,
    ):
        """
        Inflates unobserved/occluded forward sectors in the 2.5D costmap with a high
        caution cost (default: 180) instead of blindly assuming traversable free space.
        """
        if not leaves:
            return

        degraded = degraded_quadrants if degraded_quadrants is not None else self.last_degraded_quadrants
        if not degraded:
            return

        cost_to_apply = caution_cost if caution_cost is not None else self.caution_cost

        for leaf in leaves:
            angle = math_angle = np.arctan2(leaf.y, leaf.x)
            # Identify which quadrant the leaf center falls in
            if 0.0 <= math_angle < np.pi * 0.5:
                q = 0
            elif np.pi * 0.5 <= math_angle <= np.pi:
                q = 1
            elif -np.pi <= math_angle < -np.pi * 0.5:
                q = 2
            else:
                q = 3

            if q in degraded:
                if leaf.cost < cost_to_apply:
                    leaf.cost = cost_to_apply

    def preserve_terrain_memory(
        self,
        leaves: List[QuadtreeNode],
        temporal_blender: TemporalMapBlender,
    ):
        """
        Uses dead-reckoned temporal map memory to retain terrain elevations for cells
        in degraded/blind sectors where new LiDAR rays failed to return.
        """
        if not leaves or not self.last_degraded_quadrants:
            return

        for leaf in leaves:
            angle = np.arctan2(leaf.y, leaf.x)
            q = 0 if (0.0 <= angle < np.pi * 0.5) else (
                1 if (np.pi * 0.5 <= angle <= np.pi) else (
                    2 if (-np.pi <= angle < -np.pi * 0.5) else 3
                )
            )

            if q in self.last_degraded_quadrants:
                # Query historical 1D Kalman memory for this spatial cell key
                cell_key = temporal_blender._coord_to_key(leaf.x, leaf.y)
                mem = temporal_blender._cell_memory.get(cell_key)
                if mem and leaf.stats is not None:
                    # Retain last known elevation estimate
                    leaf.stats.mean_z = mem[0]
