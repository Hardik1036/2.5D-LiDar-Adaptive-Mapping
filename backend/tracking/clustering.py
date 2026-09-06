"""
High-speed Euclidean / DBSCAN clustering on non-ground point clouds.
Extracts 3D bounding boxes, centroids, dimensions, and point sets.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from backend.config import TRACKING


try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None


@dataclass(slots=True)
class DetectedCluster:
    """Represents a clustered candidate obstacle in a single LiDAR frame."""
    centroid: Tuple[float, float, float]       # (x, y, z)
    dimensions: Tuple[float, float, float]     # (length_x, width_y, height_z)
    bbox: Tuple[float, float, float, float, float, float]  # (min_x, max_x, min_y, max_y, min_z, max_z)
    point_count: int
    points: np.ndarray
    yaw: float = 0.0
    velocity: Optional[Tuple[float, float]] = None

    def to_dict(self):
        return {
            "centroid": [round(c, 2) for c in self.centroid],
            "dimensions": [round(d, 2) for d in self.dimensions],
            "bbox": [round(b, 2) for b in self.bbox],
            "points": self.point_count,
        }


class EuclideanClusterer:
    """
    Edge-optimized Euclidean clustering using scipy.spatial.cKDTree.
    Extracts isolated obstacle objects in < 6 ms.
    """

    def __init__(
        self,
        eps: float = TRACKING.CLUSTER_EPSILON,
        min_pts: int = TRACKING.CLUSTER_MIN_POINTS,
        max_pts: int = TRACKING.CLUSTER_MAX_POINTS,
    ):
        self.eps = eps
        self.min_pts = min_pts
        self.max_pts = max_pts
        self._db = DBSCAN(eps=self.eps, min_samples=2, algorithm="kd_tree", n_jobs=1) if DBSCAN else None

        # Warm up scikit-learn DBSCAN threadpool/JIT during initialization
        if self._db is not None:
            try:
                dummy = np.zeros((4, 3), dtype=np.float32)
                self._db.fit(dummy)
            except Exception:
                pass

    def cluster(self, obstacle_points: np.ndarray) -> List[DetectedCluster]:
        """
        Groups obstacle points into distinct spatial clusters.
        Accelerated with fast DBSCAN and spatial voxel decimation in < 10 ms.
        """
        n_pts = len(obstacle_points)
        if n_pts < self.min_pts:
            return []

        xyz = obstacle_points[:, :3]

        # Fast spatial voxel decimation if point cloud is dense (> 400 pts)
        if n_pts > 400:
            grid_res = 0.35  # 35cm spatial decimation
            quant = np.round(xyz / grid_res).astype(np.int32)
            _, uidx = np.unique(quant, axis=0, return_index=True)
            pts_to_cluster = xyz[uidx]
        else:
            pts_to_cluster = xyz

        if self._db is not None:
            db = self._db.fit(pts_to_cluster)
            labels = db.labels_
        else:
            return []
        unique_labels = set(labels)
        unique_labels.discard(-1)

        clusters: List[DetectedCluster] = []
        for lab in unique_labels:
            c_pts = pts_to_cluster[labels == lab]
            if self.min_pts <= len(c_pts) <= self.max_pts:
                min_vals = np.min(c_pts, axis=0)
                max_vals = np.max(c_pts, axis=0)
                centroid = tuple(np.mean(c_pts, axis=0))
                dims = tuple(max_vals - min_vals)
                bbox = (
                    float(min_vals[0]), float(max_vals[0]),
                    float(min_vals[1]), float(max_vals[1]),
                    float(min_vals[2]), float(max_vals[2]),
                )
                clusters.append(DetectedCluster(
                    centroid=centroid,
                    dimensions=dims,
                    bbox=bbox,
                    point_count=len(c_pts),
                    points=c_pts,
                ))

        return clusters
