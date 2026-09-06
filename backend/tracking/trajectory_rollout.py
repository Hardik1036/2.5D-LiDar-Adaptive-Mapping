"""
Trajectory rollout and expanding elliptical hazard cone projection.
Rolls out dynamic obstacle states forward in time (1.0s, 2.0s, 3.0s)
and computes oriented uncertainty ellipses along the heading direction.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from backend.config import ROLLOUT
from backend.tracking.kalman_tracker import TrackedObject


@dataclass(slots=True)
class HazardCone:
    """
    Oriented elliptical hazard footprint at horizon t.
    """
    time_horizon: float       # 1.0s, 2.0s, 3.0s
    center_x: float
    center_y: float
    semi_major: float         # Axis along direction of velocity (m)
    semi_minor: float         # Axis perpendicular to velocity (m)
    heading_rad: float        # Rotation angle (radians)
    polygon: List[Tuple[float, float]]  # Sampled boundary vertices for 2D visualizer

    def to_dict(self):
        return {
            "t": float(round(self.time_horizon, 1)),
            "center": [float(round(self.center_x, 2)), float(round(self.center_y, 2))],
            "semi_major": float(round(self.semi_major, 2)),
            "semi_minor": float(round(self.semi_minor, 2)),
            "heading_deg": float(round(float(np.degrees(self.heading_rad)), 1)),
            "polygon": [[float(round(p[0], 2)), float(round(p[1], 2))] for p in self.polygon],
        }


class TrajectoryRollout:
    """
    Projects dynamic obstacle kinematics into future time horizons.
    Generates expanding elliptical hazard cones accounting for position,
    velocity, and heading uncertainty.
    """

    def __init__(
        self,
        horizons: Tuple[float, float, float] = ROLLOUT.HORIZONS,
        vel_uncertainty_coeff: float = ROLLOUT.VELOCITY_UNCERTAINTY_COEFF,
        lat_uncertainty_coeff: float = ROLLOUT.LATERAL_EXPANSION_COEFF,
        min_radius: float = ROLLOUT.MIN_CONE_RADIUS,
    ):
        self.horizons = horizons
        self.vel_coeff = vel_uncertainty_coeff
        self.lat_coeff = lat_uncertainty_coeff
        self.min_radius = min_radius

    def project_object(self, track: TrackedObject) -> List[HazardCone]:
        """
        Projects future hazard cones for a single tracked dynamic obstacle.
        """
        if not track.is_dynamic:
            return []

        x0, y0 = track.x, track.y
        vx, vy = track.vx, track.vy
        speed = track.speed
        heading = track.heading
        base_len = max(track.dimensions[0], self.min_radius)
        base_width = max(track.dimensions[1], self.min_radius)

        cos_h = np.cos(heading)
        sin_h = np.sin(heading)

        cones = []
        for t in self.horizons:
            # Projected center at horizon t
            cx = x0 + vx * t
            cy = y0 + vy * t

            # Expanding uncertainty bounds
            # Major axis (longitudinal) grows with speed uncertainty
            a = (base_len / 2.0) + (self.vel_coeff * speed * t) + (0.15 * t)
            # Minor axis (lateral) grows with yaw/cross-track drift
            b = (base_width / 2.0) + (self.lat_coeff * t) + 0.10

            # Generate 16-point polygon boundary for frontend rendering
            angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            local_x = a * np.cos(angles)
            local_y = b * np.sin(angles)

            # Rotate by heading and translate to (cx, cy)
            global_x = cx + (local_x * cos_h - local_y * sin_h)
            global_y = cy + (local_x * sin_h + local_y * cos_h)

            polygon = list(zip(global_x.tolist(), global_y.tolist()))

            cones.append(HazardCone(
                time_horizon=t,
                center_x=cx,
                center_y=cy,
                semi_major=a,
                semi_minor=b,
                heading_rad=heading,
                polygon=polygon,
            ))

        return cones

    def rollout_all(self, tracks: List[TrackedObject]) -> dict:
        """
        Computes hazard projections for all active dynamic tracks.
        Returns mapping: {track_id: List[HazardCone]}.
        """
        results = {}
        for track in tracks:
            if track.is_dynamic:
                results[track.track_id] = self.project_object(track)
        return results

    def predict_hazards(self, tracks: List[TrackedObject]) -> List[dict]:
        """Returns flattened list of hazard dicts {x, y, radius, cost} for costmap inflation."""
        hazards = []
        for track in tracks:
            if track.is_dynamic:
                cones = self.project_object(track)
                for c in cones:
                    hazards.append({"x": c.center_x, "y": c.center_y, "radius": c.semi_minor, "cost": 255})
        return hazards


# Backwards-compatible alias
DynamicHazardPredictor = TrajectoryRollout
