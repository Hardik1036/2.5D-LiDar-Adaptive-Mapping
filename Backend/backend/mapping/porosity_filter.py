"""
Laser Penetration & Porosity Classifier (Feature F1.5).
Differentiates porous vegetation (tall grass, brush) from rigid structural barriers
(tree trunks, boulders, barriers) via spatial distribution and reflection intensity analysis.
"""

from typing import Any, List, Optional, Tuple
import numpy as np


class PorosityClassifier:
    """
    Analyzes cluster point penetration, height extent, and reflection intensity
    to determine whether an obstacle candidate is compliant/traversable (grass/shrub)
    or rigid/lethal (solid trunk/wall/boulder).
    """

    def __init__(
        self,
        min_veg_height: float = 0.15,
        max_veg_height: float = 1.30,
        intensity_threshold: float = 0.60,
        porous_cost: int = 20,
        rigid_cost: int = 255,
    ):
        self.min_veg_height = min_veg_height
        self.max_veg_height = max_veg_height
        self.intensity_thresh = intensity_threshold
        self.porous_cost = porous_cost
        self.rigid_cost = rigid_cost

    def evaluate_cluster(self, points: np.ndarray) -> Tuple[bool, int]:
        """
        Evaluates a cluster of 3D non-ground points.

        Args:
            points: (N, 3) or (N, 4) cluster points [x, y, z, (intensity)].

        Returns:
            Tuple (is_porous, traversability_cost):
                - (True, 20) for soft passable vegetation.
                - (False, 255) for rigid impassable obstacles.
        """
        if points is None or len(points) < 3:
            # Insufficient points to declare rigid obstacle
            return (False, self.rigid_cost)

        z = points[:, 2]
        delta_z = float(np.ptp(z)) if len(z) > 0 else 0.0

        # Porous Vegetation criteria:
        # Fast exit: if not within vegetation height bounds [0.15m, 1.30m], immediately rigid
        if not (self.min_veg_height <= delta_z <= self.max_veg_height):
            return (False, self.rigid_cost)

        # Check intensity if available in 4th column
        mean_intensity = 0.5
        if points.shape[1] >= 4:
            mean_intensity = float(np.mean(points[:, 3]))

        if mean_intensity >= 0.75:
            return (False, self.rigid_cost)
        if mean_intensity <= self.intensity_thresh:
            return (True, self.porous_cost)

        # Calculate horizontal dispersion vs vertical span for borderline intensity
        xy = points[:, :2]
        spread_xy = float(np.var(xy[:, 0]) + np.var(xy[:, 1]))
        var_z = float(np.var(z))

        if spread_xy > 0.02 and var_z > 0.005:
            return (True, self.porous_cost)

        return (False, self.rigid_cost)

    def classify_clusters(self, clusters: List[Any]) -> List[Tuple[Any, bool, int]]:
        """
        Batch evaluates a list of DetectedCluster objects.
        Returns list of tuples (cluster, is_porous, cost).
        """
        results = []
        for cluster in clusters:
            pts = getattr(cluster, "points", None)
            if pts is not None and len(pts) >= 3:
                is_porous, cost = self.evaluate_cluster(pts)
            else:
                # Fall back to cluster bounding box heuristics
                dim = getattr(cluster, "dimensions", (0.5, 0.5, 0.5))
                height = dim[2]
                if self.min_veg_height <= height <= self.max_veg_height:
                    is_porous, cost = (True, self.porous_cost)
                else:
                    is_porous, cost = (False, self.rigid_cost)
            results.append((cluster, is_porous, cost))
        return results
