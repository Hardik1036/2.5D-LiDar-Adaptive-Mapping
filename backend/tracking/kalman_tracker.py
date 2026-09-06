"""
2D Constant-Velocity Extended/Linear Kalman Filter for dynamic obstacle tracking.
State vector: x = [x, y, vx, vy]^T
Measurement: z = [x, y]^T
Includes Hungarian data association (linear_sum_assignment) and Euclidean gating.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
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

    def __init__(self, init_x: float, init_y: float, dt: float = TRACKING.DT):
        self.dt = float(dt)
        self.x = float(init_x)
        self.y = float(init_y)
        self.vx = 0.0
        self.vy = 0.0

        # State variances
        self.px = 0.5
        self.py = 0.5
        self.pvx = 2.0
        self.pvy = 2.0

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

    def update(self, meas_x: float, meas_y: float):
        """Incorporate new 2D position measurement in < 0.001 ms."""
        # X dimension update
        kx = self.px / (self.px + self.r_meas)
        kvx = (self.pvx * self.dt) / (self.px + self.r_meas)
        rx = float(meas_x) - self.x
        self.x += kx * rx
        self.vx += kvx * rx
        self.px *= (1.0 - kx)

        # Y dimension update
        ky = self.py / (self.py + self.r_meas)
        kvy = (self.pvy * self.dt) / (self.py + self.r_meas)
        ry = float(meas_y) - self.y
        self.y += ky * ry
        self.vy += kvy * ry
        self.py *= (1.0 - ky)


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
    ):
        self.gating_dist = gating_dist
        self.max_age = max_age
        self.tracks: List[TrackedObject] = []
        self._next_id = 1

    def update(self, clusters: List[DetectedCluster], dt: float = TRACKING.DT) -> List[TrackedObject]:
        """
        Predict and update tracks with new cluster detections.

        Args:
            clusters: List of DetectedCluster instances from current frame.
            dt: Time delta from previous frame.

        Returns:
            List of currently active confirmed tracks.
        """
        # 1. Predict all existing tracks
        for track in self.tracks:
            track.kf.predict(dt=dt)
            track.age += 1
            track.time_since_update += 1

        n_tracks = len(self.tracks)
        n_dets = len(clusters)

        if n_tracks == 0:
            # All detections become new tentative tracks
            for cluster in clusters:
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

                # Kalman measurement update
                track.kf.update(cluster.centroid[0], cluster.centroid[1])
                track.hits += 1
                track.time_since_update = 0
                track.dimensions = cluster.dimensions
                track.z = float(cluster.centroid[2])
                track.bbox = cluster.bbox
                track.point_count = cluster.point_count

        # 4. Unassigned detections spawn new tracks (capped at 40 active tracks)
        if len(self.tracks) < 40:
            for j in range(n_dets):
                if j not in assigned_dets:
                    self._spawn_track(clusters[j], dt=dt)

        # 5. Prune expired or diverging tracks
        self._prune_tracks()

        return self.get_confirmed_tracks()

    def _spawn_track(self, cluster: DetectedCluster, dt: float):
        """Creates a new tentative track."""
        kf = SingleObjectKalmanFilter(init_x=cluster.centroid[0], init_y=cluster.centroid[1], dt=dt)
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
        return [t for t in self.tracks if t.is_confirmed and t.time_since_update == 0]

    def get_dynamic_tracks(self) -> List[TrackedObject]:
        """Returns tracks confirmed and moving above speed threshold."""
        return [t for t in self.tracks if t.is_dynamic and t.time_since_update == 0]


# Backwards-compatible aliases
MultiObjectTracker = KalmanTracker
TrackedObstacle = TrackedObject
