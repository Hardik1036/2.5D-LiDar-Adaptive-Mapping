"""
2D Constant-Velocity Extended/Linear Kalman Filter for dynamic obstacle tracking.
State vector: x = [x, y, vx, vy]^T
Measurement: z = [x, y]^T
Includes Hungarian data association (linear_sum_assignment) and Euclidean gating.
"""

import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import linear_sum_assignment

from backend.config import TRACKING
from backend.tracking.clustering import DetectedCluster


class SingleObjectKalmanFilter:
    """
    Edge-optimized 2D Constant-Velocity Kalman Filter for an individual track.
    Decoupled X and Y kinematics for sub-microsecond state estimation.
    """
    __slots__ = ("x", "y", "vx", "vy", "px", "py", "pvx", "pvy", "dt", "q_pos", "q_vel", "r_meas")

    def __init__(
        self,
        init_x: float,
        init_y: float,
        dt: float = TRACKING.DT,
        init_vx: float = 0.0,
        init_vy: float = 0.0,
    ):
        self.dt = float(dt)
        self.x = float(init_x)
        self.y = float(init_y)
        self.vx = float(init_vx) if (init_vx is not None and np.isfinite(init_vx)) else 0.0
        self.vy = float(init_vy) if (init_vy is not None and np.isfinite(init_vy)) else 0.0

        # State variances
        self.px = 0.5
        self.py = 0.5
        self.pvx = 2.0 if (self.vx == 0.0 and self.vy == 0.0) else 1.0
        self.pvy = 2.0 if (self.vx == 0.0 and self.vy == 0.0) else 1.0

        # Noise parameters
        self.q_pos = 0.05
        self.q_vel = 0.20
        self.r_meas = 0.08

    @property
    def P(self) -> np.ndarray:
        """Compatibility property returning 4x4 covariance matrix."""
        return np.diag([self.px, self.py, self.pvx, self.pvy])

    def predict(self, dt: Optional[float] = None):
        """Predict state and variance forward in < 0.001 ms."""
        step = dt if (dt is not None and dt > 0) else self.dt
        self.x += self.vx * step
        self.y += self.vy * step

        # Variance propagation
        step2 = step * step
        self.px += self.pvx * step2 + self.q_pos
        self.py += self.pvy * step2 + self.q_pos
        self.pvx += self.q_vel
        self.pvy += self.q_vel

    def update(
        self,
        meas_x: float,
        meas_y: float,
        meas_vx: Optional[float] = None,
        meas_vy: Optional[float] = None,
    ):
        """Incorporate new 2D position measurement in < 0.001 ms."""
        # X dimension update
        kx = self.px / (self.px + self.r_meas)
        kvx = (self.pvx * self.dt) / (self.px + self.r_meas)
        rx = float(meas_x) - self.x
        self.x += kx * rx
        self.vx += kvx * rx
        self.px *= (1.0 - kx)
        self.pvx = max(0.05, self.pvx * max(0.1, 1.0 - kvx * self.dt))

        # Y dimension update
        ky = self.py / (self.py + self.r_meas)
        kvy = (self.pvy * self.dt) / (self.py + self.r_meas)
        ry = float(meas_y) - self.y
        self.y += ky * ry
        self.vy += kvy * ry
        self.py *= (1.0 - ky)
        self.pvy = max(0.05, self.pvy * max(0.1, 1.0 - kvy * self.dt))

        # Optional direct velocity blending if available
        if meas_vx is not None and np.isfinite(meas_vx):
            self.vx = 0.7 * self.vx + 0.3 * float(meas_vx)
        if meas_vy is not None and np.isfinite(meas_vy):
            self.vy = 0.7 * self.vy + 0.3 * float(meas_vy)


@dataclass
class TrackedObject:
    """Represents an active tracked entity across consecutive frames."""
    track_id: int
    kf: SingleObjectKalmanFilter
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    dimensions: Tuple[float, float, float] = (0.5, 0.5, 1.0)
    z: float = 0.0
    bbox: Tuple[float, float, float, float, float, float] = (0, 0, 0, 0, 0, 0)
    point_count: int = 0

    @property
    def x(self) -> float:
        return self.kf.x

    @property
    def y(self) -> float:
        return self.kf.y

    @property
    def vx(self) -> float:
        return self.kf.vx

    @property
    def vy(self) -> float:
        return self.kf.vy

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))

    @property
    def heading(self) -> float:
        """Heading angle theta in radians [-pi, pi]."""
        return float(np.arctan2(self.vy, self.vx))

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= TRACKING.MIN_HITS_TO_CONFIRM

    @property
    def is_dynamic(self) -> bool:
        return self.is_confirmed and (self.speed >= TRACKING.DYNAMIC_SPEED_THRESHOLD)

    def to_dict(self):
        return {
            "id": int(self.track_id),
            "position": [float(round(self.x, 2)), float(round(self.y, 2)), float(round(self.z, 2))],
            "velocity": [float(round(self.vx, 2)), float(round(self.vy, 2))],
            "speed": float(round(self.speed, 2)),
            "heading": float(round(self.heading, 3)),
            "heading_deg": float(round(np.degrees(self.heading), 1)),
            "dimensions": [float(round(d, 2)) for d in self.dimensions],
            "is_dynamic": bool(self.is_dynamic),
            "is_confirmed": bool(self.is_confirmed),
            "age": int(self.age),
            "hits": int(self.hits),
            "point_count": int(self.point_count),
        }


class KalmanTracker:
    """
    Multi-object 2D Kalman Tracker.
    Associates cluster detections to existing tracks via Hungarian algorithm,
    maintains track lifecycle, and filters spurious observations.
    """

    def __init__(
        self,
        gating_dist: float = TRACKING.GATING_DISTANCE,
        max_age: int = TRACKING.MAX_AGE_BEFORE_DELETION,
        max_tracks: int = getattr(TRACKING, "MAX_TRACKS", 40),
    ):
        self.gating_dist = gating_dist
        self.max_age = max_age
        self.max_tracks = max_tracks
        self.tracks: List[TrackedObject] = []
        self._next_id = 1
        self._lock = threading.RLock()

    def update(
        self,
        clusters: Union[List[DetectedCluster], np.ndarray],
        dt: float = TRACKING.DT,
    ) -> List[TrackedObject]:
        """
        Predict and update tracks with new cluster detections or (M, 9)/(M, >=6) ndarrays.

        Args:
            clusters: List of DetectedCluster instances or (M, 9) ndarray from current frame.
            dt: Time delta from previous frame.

        Returns:
            List of currently active confirmed tracks.
        """
        with self._lock:
            # Handle numpy array proposal inputs directly
            if isinstance(clusters, np.ndarray):
                if clusters.ndim == 1:
                    clusters = clusters.reshape(1, -1) if len(clusters) > 0 else np.empty((0, 9))
                cluster_list: List[DetectedCluster] = []
                for det in clusters:
                    if len(det) >= 6:
                        x, y, z = float(det[0]), float(det[1]), float(det[2])
                        l, w, h = float(det[3]), float(det[4]), float(det[5])
                        yaw = float(det[6]) if len(det) >= 7 else 0.0
                        vel = (float(det[7]), float(det[8])) if len(det) >= 9 else None
                        min_x, max_x = x - l * 0.5, x + l * 0.5
                        min_y, max_y = y - w * 0.5, y + w * 0.5
                        min_z, max_z = z - h * 0.5, z + h * 0.5
                        cluster_list.append(
                            DetectedCluster(
                                centroid=(x, y, z),
                                dimensions=(l, w, h),
                                bbox=(min_x, max_x, min_y, max_y, min_z, max_z),
                                point_count=50,
                                points=np.empty((0, 3), dtype=np.float32),
                                yaw=yaw,
                                velocity=vel,
                            )
                        )
                clusters = cluster_list

            # 1. Predict all existing tracks
            for track in self.tracks:
                track.kf.predict(dt=dt)
                track.age += 1
                track.time_since_update += 1

            n_tracks = len(self.tracks)
            n_dets = len(clusters)

            if n_tracks == 0:
                # All detections become new tentative tracks up to max_tracks
                for cluster in clusters:
                    if len(self.tracks) >= self.max_tracks:
                        break
                    self._spawn_track(cluster, dt=dt)
                return self.get_confirmed_tracks()

            if n_dets == 0:
                # No detections this frame, prune expired tracks
                self._prune_tracks()
                return self.get_confirmed_tracks()

            from scipy.spatial.distance import cdist

            # 2. Build cost matrix via vectorized cdist (Euclidean distance on XY plane)
            track_coords = np.array([[t.x, t.y] for t in self.tracks], dtype=np.float64)
            det_coords = np.array([c.centroid[:2] for c in clusters], dtype=np.float64)
            cost_matrix = cdist(track_coords, det_coords)

            # 3. Hungarian association
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            assigned_tracks = set()
            assigned_dets = set()

            for r, c in zip(row_indices, col_indices):
                if cost_matrix[r, c] <= self.gating_dist:
                    assigned_tracks.add(r)
                    assigned_dets.add(c)
                    track = self.tracks[r]
                    cluster = clusters[c]

                    # Kalman measurement update (position and optional velocity)
                    vel = cluster.velocity
                    vx_meas = vel[0] if vel is not None else None
                    vy_meas = vel[1] if vel is not None else None
                    track.kf.update(cluster.centroid[0], cluster.centroid[1], meas_vx=vx_meas, meas_vy=vy_meas)
                    track.hits += 1
                    track.time_since_update = 0
                    track.dimensions = cluster.dimensions
                    track.z = float(cluster.centroid[2])
                    track.bbox = cluster.bbox
                    track.point_count = cluster.point_count

            # 4. Prune expired or diverging tracks before calculating available capacity
            self._prune_tracks()

            # 5. Unassigned detections spawn new tracks (strictly capped at max_tracks)
            for j in range(n_dets):
                if len(self.tracks) >= self.max_tracks:
                    break
                if j not in assigned_dets:
                    self._spawn_track(clusters[j], dt=dt)

            return self.get_confirmed_tracks()

    def _spawn_track(self, cluster: DetectedCluster, dt: float):
        """Creates a new tentative track."""
        if len(self.tracks) >= self.max_tracks:
            return
        init_vx = (
            float(cluster.velocity[0])
            if cluster.velocity is not None and len(cluster.velocity) > 0 and np.isfinite(cluster.velocity[0])
            else 0.0
        )
        init_vy = (
            float(cluster.velocity[1])
            if cluster.velocity is not None and len(cluster.velocity) > 1 and np.isfinite(cluster.velocity[1])
            else 0.0
        )
        kf = SingleObjectKalmanFilter(
            init_x=cluster.centroid[0],
            init_y=cluster.centroid[1],
            dt=dt,
            init_vx=init_vx,
            init_vy=init_vy,
        )
        track = TrackedObject(
            track_id=self._next_id,
            kf=kf,
            hits=1,
            age=1,
            time_since_update=0,
            dimensions=cluster.dimensions,
            z=cluster.centroid[2],
            bbox=cluster.bbox,
            point_count=cluster.point_count,
        )
        self._next_id += 1
        self.tracks.append(track)

    def _prune_tracks(self):
        """Removes tracks that have exceeded max_age or covariance limits."""
        valid_tracks = []
        for track in self.tracks:
            # Check age
            if track.time_since_update <= self.max_age:
                # Check covariance trace to prevent divergence
                cov_trace = float(np.trace(track.kf.P))
                if cov_trace < TRACKING.MAX_COVARIANCE_TRACE:
                    valid_tracks.append(track)
        self.tracks = valid_tracks

    def get_confirmed_tracks(self) -> List[TrackedObject]:
        """Returns only confirmed active tracks."""
        with self._lock:
            return [t for t in self.tracks if t.is_confirmed and t.time_since_update == 0]

    def get_dynamic_tracks(self) -> List[TrackedObject]:
        """Returns tracks confirmed and moving above speed threshold."""
        with self._lock:
            return [t for t in self.tracks if t.is_dynamic and t.time_since_update == 0]


# Backwards-compatible aliases
MultiObjectTracker = KalmanTracker
TrackedObstacle = TrackedObject

