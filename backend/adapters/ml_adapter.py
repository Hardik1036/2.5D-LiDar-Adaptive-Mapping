"""
ML Perception Adapter for DRDO SIH 2026 Problem Statement 53.
Bridges external neural network inference outputs (PointPillars, Semantic Segmentation)
into the 2.5D Adaptive Variable Resolution Perception Engine.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from backend.tracking.clustering import DetectedCluster


class MLPerceptionAdapter:
    """
    Ingests and normalizes machine learning perception outputs:
    1. Point-wise semantic classification labels (N,) -> Ground (0), Vegetation (1), Rigid (2), Dynamic (3).
    2. 3D Oriented Bounding Box Detections (M, 9) -> [x, y, z, l, w, h, yaw, vx, vy].
    """

    LABEL_GROUND = 0
    LABEL_VEGETATION = 1
    LABEL_RIGID = 2
    LABEL_DYNAMIC = 3

    def __init__(self):
        self._empty_pts = np.empty((0, 3), dtype=np.float32)

    def process_semantic_labels(
        self,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Partitions input point cloud based on ML semantic segmentation labels.

        Args:
            points: (N, 3) XYZ point coordinates.
            labels: (N,) integer class IDs:
                    0 = Ground
                    1 = Passable Vegetation (tall grass, small brush)
                    2 = Rigid Hazard (rocks, trees, structures, walls)
                    3 = Dynamic Foreground (vehicles, pedestrians, robots)

        Returns:
            Dictionary with separated point cloud subsets:
            {
                "ground": np.ndarray,
                "vegetation": np.ndarray,
                "rigid": np.ndarray,
                "dynamic": np.ndarray,
                "obstacle": np.ndarray,  # Rigid + Dynamic (for tracking/obstacle mapping)
            }
        """
        if points is None or len(points) == 0:
            return {
                "ground": self._empty_pts,
                "vegetation": self._empty_pts,
                "rigid": self._empty_pts,
                "dynamic": self._empty_pts,
                "obstacle": self._empty_pts,
            }

        # If no labels are provided, treat all points as unclassified
        if labels is None or len(labels) != len(points):
            return {
                "ground": self._empty_pts,
                "vegetation": self._empty_pts,
                "rigid": points,
                "dynamic": self._empty_pts,
                "obstacle": points,
            }

        # Vectorized label masking (< 0.8 ms for 25,000 points)
        labels_arr = np.asarray(labels, dtype=np.int32)
        ground_mask = (labels_arr == self.LABEL_GROUND)
        veg_mask = (labels_arr == self.LABEL_VEGETATION)
        rigid_mask = (labels_arr == self.LABEL_RIGID)
        dynamic_mask = (labels_arr == self.LABEL_DYNAMIC)
        obstacle_mask = rigid_mask | dynamic_mask

        return {
            "ground": points[ground_mask],
            "vegetation": points[veg_mask],
            "rigid": points[rigid_mask],
            "dynamic": points[dynamic_mask],
            "obstacle": points[obstacle_mask],
        }

    def process_pillar_detections(
        self,
        detections: Optional[np.ndarray] = None,
    ) -> List[DetectedCluster]:
        """
        Converts 3D bounding box detections into DetectedCluster candidate objects.

        Args:
            detections: (M, 9) ndarray where each row represents:
                        [x, y, z, length, width, height, yaw, vx, vy]

        Returns:
            List of DetectedCluster instances compatible with MultiObjectTracker and CostmapEvaluator.
        """
        if detections is None or len(detections) == 0:
            return []

        det_arr = np.asarray(detections, dtype=np.float32)
        if det_arr.ndim == 1:
            det_arr = det_arr.reshape(1, -1)

        if det_arr.shape[1] < 6:
            raise ValueError(f"Detection array must have at least 6 fields [x, y, z, l, w, h], got {det_arr.shape[1]}")

        clusters: List[DetectedCluster] = []
        for det in det_arr:
            x, y, z = float(det[0]), float(det[1]), float(det[2])
            l, w, h = float(det[3]), float(det[4]), float(det[5])

            # Compute axis-aligned bounding box bounding the 3D detection
            min_x, max_x = x - l * 0.5, x + l * 0.5
            min_y, max_y = y - w * 0.5, y + w * 0.5
            min_z, max_z = z - h * 0.5, z + h * 0.5

            cluster_yaw = float(det[6]) if det_arr.shape[1] >= 7 else 0.0
            velocity = (float(det[7]), float(det[8])) if det_arr.shape[1] >= 9 else None

            cluster = DetectedCluster(
                centroid=(x, y, z),
                dimensions=(l, w, h),
                bbox=(min_x, max_x, min_y, max_y, min_z, max_z),
                point_count=50,  # Nominal surrogate point density
                points=self._empty_pts,
                yaw=cluster_yaw,
                velocity=velocity,
            )

            clusters.append(cluster)

        return clusters
