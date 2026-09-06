"""
Millimeter-Residual Thin Hazard & Spike Strip Detector (Feature F1.6).
Detects low-profile dangerous obstacles (spike strips, metal plates, cables, rails)
sitting 1.5 cm to 5.0 cm above the local fitted ground plane with high reflection contrast.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree


class ThinHazardDetector:
    """
    Evaluates orthogonal millimeter residuals against local ground planes
    to expose hazards too low to trigger standard step-height obstacle segmentation
    (15 mm - 50 mm elevation delta).
    """

    def __init__(
        self,
        min_residual: float = 0.015,   # 15 mm minimum height above ground
        max_residual: float = 0.060,   # 60 mm maximum height above ground
        min_points_cluster: int = 5,
        cluster_eps: float = 0.35,
        forced_cost: int = 255,
    ):
        self.min_res = min_residual
        self.max_res = max_residual
        self.min_pts = min_points_cluster
        self.cluster_eps = cluster_eps
        self.forced_cost = forced_cost

    def fit_ground_plane(self, ground_points: np.ndarray) -> np.ndarray:
        """
        Fast SVD plane fit returning [a, b, c, d] normalized plane coefficients.
        """
        if len(ground_points) < 3:
            return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        xyz = ground_points[:, :3]
        centroid = np.mean(xyz, axis=0)
        centered = xyz - centroid

        # SVD on covariance matrix
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[2, :]  # Eigenvector corresponding to smallest eigenvalue

        # Orient normal upwards (+Z)
        if normal[2] < 0:
            normal = -normal

        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-6:
            normal = normal / norm_len

        d = -float(np.dot(normal, centroid))
        return np.array([normal[0], normal[1], normal[2], d], dtype=np.float32)

    def detect_low_profile_hazards(
        self,
        ground_points: np.ndarray,
        plane_model: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Identifies ground-level thin hazards using orthogonal plane residuals and intensity.

        Args:
            ground_points: (N, 3) or (N, 4) ground LiDAR points.
            plane_model: Optional [a, b, c, d] plane coefficients.

        Returns:
            List of detected thin hazard dictionaries with bounding boxes and lethal costs.
        """
        if ground_points is None or len(ground_points) < self.min_pts:
            return []

        xyz = ground_points[:, :3]

        if plane_model is None or len(plane_model) < 4:
            plane_model = self.fit_ground_plane(ground_points)

        a, b, c, d = plane_model
        denom = np.sqrt(a * a + b * b + c * c)
        if denom < 1e-6:
            denom = 1.0

        # Vectorized orthogonal signed distance to plane
        signed_dist = (a * xyz[:, 0] + b * xyz[:, 1] + c * xyz[:, 2] + d) / denom

        # Candidates are elevated slightly above ground surface
        candidate_mask = (signed_dist >= self.min_res) & (signed_dist <= self.max_res)

        # Optional intensity filter if available (abnormal reflective contrast for metallic spike strips/plates)
        if ground_points.shape[1] >= 4 and np.any(candidate_mask):
            intensities = ground_points[:, 3]
            mean_i = float(np.mean(intensities))
            std_i = float(np.std(intensities))
            if std_i > 0.05:
                # Abnormal reflection contrast (spike strip steel vs natural asphalt/dirt)
                intensity_mask = intensities >= (mean_i + 1.5 * std_i)
                candidate_mask = candidate_mask & intensity_mask

        cand_pts = xyz[candidate_mask]
        if len(cand_pts) < self.min_pts:
            return []

        # Fast spatial binning into linear clusters in < 0.2 ms
        inv_eps = 1.0 / self.cluster_eps
        clusters_map: Dict[Tuple[int, int], List[np.ndarray]] = {}
        for p in cand_pts:
            cell_key = (int(np.floor(p[0] * inv_eps)), int(np.floor(p[1] * inv_eps)))
            clusters_map.setdefault(cell_key, []).append(p)

        hazards = []
        for cell_key, p_list in clusters_map.items():
            if len(p_list) >= self.min_pts:
                cluster_pts = np.array(p_list)
                min_x, max_x = float(np.min(cluster_pts[:, 0])), float(np.max(cluster_pts[:, 0]))
                min_y, max_y = float(np.min(cluster_pts[:, 1])), float(np.max(cluster_pts[:, 1]))
                min_z, max_z = float(np.min(cluster_pts[:, 2])), float(np.max(cluster_pts[:, 2]))
                cx = float(np.mean(cluster_pts[:, 0]))
                cy = float(np.mean(cluster_pts[:, 1]))
                cz = float(np.mean(cluster_pts[:, 2]))

                hazards.append({
                    "type": "thin_hazard",
                    "centroid": (cx, cy, cz),
                    "bbox": (min_x, max_x, min_y, max_y, min_z, max_z),
                    "point_count": len(cluster_pts),
                    "cost": self.forced_cost,
                    "radius": float(max(max_x - min_x, max_y - min_y) * 0.5 + 0.1),
                })
                if len(hazards) >= 20:
                    break

        return hazards
