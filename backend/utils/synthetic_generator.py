"""
High-speed programmatic synthetic point cloud generator for 2.5D LiDAR perception.
Simulates:
  - Ground plane with slopes, undulations, curbs, and roughness.
  - Static structures (walls, curbs, rocks).
  - Dynamic obstacles (moving pedestrian and moving rover/vehicle) with kinematics.
  - LiDAR range limits and Gaussian beam noise.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from backend.config import BOUNDS


@dataclass
class DynamicObjectSim:
    """State of a simulated moving object."""
    obj_id: int
    name: str
    x: float
    y: float
    vx: float
    vy: float
    width: float
    length: float
    height: float
    points_per_frame: int = 150
    trajectory_type: str = "linear"  # "linear" or "circular"
    circle_center: Tuple[float, float] = (0.0, 0.0)
    circle_radius: float = 6.0
    circle_omega: float = 0.25  # rad/s


class SyntheticLiDARGenerator:
    """
    Generates realistic 2.5D / 3D LiDAR point clouds at >= 25 Hz.
    Point array shape: (N, 4) with columns [x, y, z, intensity].
    """

    def __init__(self, seed: Optional[int] = 42, sensor_height: float = 1.6):
        self.rng = np.random.default_rng(seed)
        self.sensor_height = sensor_height
        self.ground_z = -sensor_height  # Ground is at z = -1.6m if sensor at origin

        # Initialize dynamic actors
        self.actors: List[DynamicObjectSim] = [
            # Pedestrian moving along Y axis
            DynamicObjectSim(
                obj_id=1,
                name="pedestrian",
                x=4.0,
                y=-10.0,
                vx=0.0,
                vy=1.2,  # 1.2 m/s
                width=0.5,
                length=0.5,
                height=1.75,
                points_per_frame=120,
                trajectory_type="linear",
            ),
            # Autonomous rover/vehicle doing circular traversal
            DynamicObjectSim(
                obj_id=2,
                name="vehicle",
                x=8.0,
                y=0.0,
                vx=0.0,
                vy=3.0,
                width=1.8,
                length=3.2,
                height=1.4,
                points_per_frame=350,
                trajectory_type="circular",
                circle_center=(0.0, 2.0),
                circle_radius=8.0,
                circle_omega=0.35,  # rad/s
            ),
        ]

        # Pre-generate static ground grid to accelerate real-time runtime (< 5ms)
        self._build_static_scene()

    def _build_static_scene(self):
        """Constructs base ground mesh and static obstacles."""
        # Concentric ring scan pattern matching standard 16/32-beam edge LiDAR (10.8k pts)
        radii = np.linspace(0.8, 19.5, 60)
        angles = np.linspace(-np.pi, np.pi, 180, endpoint=False)
        r_grid, a_grid = np.meshgrid(radii, angles)

        x_pts = (r_grid * np.cos(a_grid)).flatten()
        y_pts = (r_grid * np.sin(a_grid)).flatten()

        # Elevation equation: gentle slope in X, sinusoidal undulation in Y
        z_pts = np.full_like(x_pts, self.ground_z)
        z_pts += 0.04 * x_pts  # Gentle 2.3 deg slope along X
        z_pts += 0.06 * np.sin(0.4 * y_pts)  # Terrain undulation

        # Add a raised curb step along a road edge (x in [5.8, 6.1], y in [-5.0, 5.0])
        curb_mask = (x_pts >= 5.8) & (x_pts <= 6.1) & (y_pts >= -5.0) & (y_pts <= 5.0)
        z_pts[curb_mask] += 0.18

        # Base ground intensity
        intensities = np.clip(0.3 + 0.1 * np.cos(x_pts), 0.1, 0.9)

        self.static_ground = np.column_stack([x_pts, y_pts, z_pts, intensities])

        # Add static obstacles (e.g. concrete barrier / wall)
        wall_x = np.linspace(-8.0, -8.0, 2)
        wall_y = np.linspace(-4.0, 4.0, 35)
        wall_z = np.linspace(self.ground_z, self.ground_z + 1.2, 8)
        wx, wy, wz = np.meshgrid(wall_x, wall_y, wall_z)
        wall_points = np.column_stack([
            wx.flatten(),
            wy.flatten(),
            wz.flatten(),
            np.full(wx.size, 0.85),
        ])

        # Second static obstacle: boulder/pillar
        pillar_x = np.linspace(2.0, 2.8, 6)
        pillar_y = np.linspace(6.0, 6.8, 6)
        pillar_z = np.linspace(self.ground_z, self.ground_z + 0.8, 6)
        px, py, pz = np.meshgrid(pillar_x, pillar_y, pillar_z)
        pillar_points = np.column_stack([
            px.flatten(),
            py.flatten(),
            pz.flatten(),
            np.full(px.size, 0.70),
        ])

        self.static_obstacles = np.vstack([wall_points, pillar_points])

    def update_actors(self, dt: float, timestamp: float):
        """Advance kinematics of dynamic obstacles."""
        for actor in self.actors:
            if actor.trajectory_type == "linear":
                actor.x += actor.vx * dt
                actor.y += actor.vy * dt

                # Reverse when reaching boundary
                if actor.y > BOUNDS.Y_MAX - 3.0:
                    actor.vy = -abs(actor.vy)
                elif actor.y < BOUNDS.Y_MIN + 3.0:
                    actor.vy = abs(actor.vy)

            elif actor.trajectory_type == "circular":
                cx, cy = actor.circle_center
                angle = actor.circle_omega * timestamp
                actor.x = cx + actor.circle_radius * np.cos(angle)
                actor.y = cy + actor.circle_radius * np.sin(angle)
                # Tangential velocity
                actor.vx = -actor.circle_radius * actor.circle_omega * np.sin(angle)
                actor.vy = actor.circle_radius * actor.circle_omega * np.cos(angle)

    def _sample_box_points(self, actor: DynamicObjectSim) -> np.ndarray:
        """Sample LiDAR surface points on a dynamic 3D bounding box."""
        n_pts = actor.points_per_frame
        dx = self.rng.uniform(-actor.length / 2, actor.length / 2, n_pts)
        dy = self.rng.uniform(-actor.width / 2, actor.width / 2, n_pts)
        dz = self.rng.uniform(0.05, actor.height, n_pts)

        x = actor.x + dx
        y = actor.y + dy
        z = self.ground_z + dz
        intensity = self.rng.uniform(0.6, 1.0, n_pts)

        return np.column_stack([x, y, z, intensity])

    def generate_frame(self, timestamp: float = 0.0, dt: float = 0.04, t: Optional[float] = None) -> Tuple[np.ndarray, List[dict]]:
        """
        Produce a full frame point cloud at the given timestamp.
        Returns:
            points: np.ndarray of shape (N, 4) [x, y, z, intensity]
            ground_truth_actors: list of dicts with current actor truth
        """
        if t is not None:
            timestamp = t
        self.update_actors(dt=dt, timestamp=timestamp)

        # 1. Base ground with subtle jitter/measurement noise (sigma = 0.015m)
        noise = self.rng.normal(0.0, 0.012, size=(self.static_ground.shape[0], 3))
        ground_noisy = self.static_ground.copy()
        ground_noisy[:, :3] += noise

        # 2. Static obstacles
        static_obs = self.static_obstacles.copy()
        static_obs[:, :3] += self.rng.normal(0.0, 0.01, size=(static_obs.shape[0], 3))

        # 3. Dynamic obstacles
        dynamic_point_list = []
        gt_metadata = []
        for actor in self.actors:
            pts = self._sample_box_points(actor)
            dynamic_point_list.append(pts)
            gt_metadata.append({
                "id": actor.obj_id,
                "name": actor.name,
                "position": [float(actor.x), float(actor.y), float(self.ground_z + actor.height / 2)],
                "velocity": [float(actor.vx), float(actor.vy)],
                "speed": float(np.hypot(actor.vx, actor.vy)),
                "heading": float(np.arctan2(actor.vy, actor.vx)),
                "dimensions": [float(actor.length), float(actor.width), float(actor.height)],
            })

        all_points = np.vstack([ground_noisy, static_obs] + dynamic_point_list)

        # Filter strictly to spatial bounds
        mask = (
            (all_points[:, 0] >= BOUNDS.X_MIN) & (all_points[:, 0] <= BOUNDS.X_MAX) &
            (all_points[:, 1] >= BOUNDS.Y_MIN) & (all_points[:, 1] <= BOUNDS.Y_MAX) &
            (all_points[:, 2] >= BOUNDS.Z_MIN) & (all_points[:, 2] <= BOUNDS.Z_MAX)
        )
        points = all_points[mask].astype(np.float32)

        return points, gt_metadata


SyntheticScanGenerator = SyntheticLiDARGenerator
