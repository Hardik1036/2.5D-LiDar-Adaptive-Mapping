"""
Global configuration constants and parameters for SIH 2026 Problem Statement 53:
DRDO: Adaptive Variable Resolution 2.5D LiDAR Mapping for Dynamic Perception.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class SpatialBounds:
    """Operational 3D bounding box for LiDAR perception (meters)."""
    X_MIN: float = -20.0
    X_MAX: float = 20.0
    Y_MIN: float = -20.0
    Y_MAX: float = 20.0
    Z_MIN: float = -3.0
    Z_MAX: float = 3.0

    @property
    def x_range(self) -> float:
        return self.X_MAX - self.X_MIN

    @property
    def y_range(self) -> float:
        return self.Y_MAX - self.Y_MIN

    @property
    def z_range(self) -> float:
        return self.Z_MAX - self.Z_MIN


@dataclass(frozen=True)
class QuadtreeConfig:
    """Adaptive 2.5D Quadtree spatial partitioning configuration."""
    COARSE_RESOLUTION: float = 0.50  # 50 cm coarse cell size
    FINE_RESOLUTION: float = 0.05    # 5 cm fine cell size (leaf minimum)
    
    # Adaptive subdivision criteria:
    # Subdivide if internal point height variance > TAU_SIGMA or delta_z > TAU_Z
    TAU_SIGMA: float = 0.04          # Variance threshold (m^2)
    TAU_Z: float = 0.12              # Step height / elevation delta threshold (m)
    
    # Minimum point count required in a node to justify subdivision
    MIN_POINTS_PER_CELL: int = 3
    # Max depth calculated dynamically based on coarse -> fine resolutions


@dataclass(frozen=True)
class CostmapConfig:
    """Traversability cost values and evaluation thresholds."""
    # Cost bounds
    SAFE_MAX: int = 50               # 0 - 50 = Safe traversable terrain
    CAUTION_MAX: int = 180           # 51 - 180 = Caution / steep slope
    LETHAL_VAL: int = 255            # 181 - 255 = Lethal obstacle / non-traversable
    
    # Slope thresholds in degrees
    MAX_TRAVERSABLE_SLOPE_DEG: float = 15.0
    LETHAL_SLOPE_DEG: float = 30.0
    
    # Obstacle step height threshold in meters
    MAX_STEP_HEIGHT: float = 0.15
    LETHAL_STEP_HEIGHT: float = 0.25


@dataclass(frozen=True)
class GroundSegmentationConfig:
    """Ground vs non-ground point cloud segmentation parameters."""
    NUM_RADIAL_BINS: int = 16
    NUM_ANGULAR_SECTORS: int = 32
    MAX_GROUND_SLOPE_DEG: float = 12.0
    SENSOR_HEIGHT_NOMINAL: float = 1.6  # Typical sensor mount height above ground (m)
    GROUND_DISTANCE_THRESHOLD: float = 0.12  # Distance to estimated ground plane
    RANSAC_MAX_ITERATIONS: int = 40


@dataclass(frozen=True)
class TrackingConfig:
    """Clustering, 2D Extended Kalman Filter, and association parameters."""
    # DBSCAN / Euclidean clustering
    CLUSTER_EPSILON: float = 0.45       # Neighborhood radius (m)
    CLUSTER_MIN_POINTS: int = 4         # Min points for cluster candidate
    CLUSTER_MAX_POINTS: int = 1500
    
    # Kalman Filter & Data Association
    DT: float = 0.04                    # Nominal time step at 25 Hz (seconds)
    GATING_DISTANCE: float = 1.5        # Euclidean gating distance threshold (m)
    MAX_COVARIANCE_TRACE: float = 10.0  # Threshold to declare track diverging
    
    # Track lifecycle
    MIN_HITS_TO_CONFIRM: int = 3        # Number of frames to confirm track
    MAX_AGE_BEFORE_DELETION: int = 5    # Missed frames before deleting track
    
    # Dynamic obstacle velocity threshold to qualify for dynamic hazard prediction
    DYNAMIC_SPEED_THRESHOLD: float = 0.25  # m/s


@dataclass(frozen=True)
class RolloutConfig:
    """Trajectory rollout and expanding hazard cone settings."""
    HORIZONS: Tuple[float, float, float] = (1.0, 2.0, 3.0)  # Projection horizons (seconds)
    VELOCITY_UNCERTAINTY_COEFF: float = 0.15                # Forward cone expansion factor
    LATERAL_EXPANSION_COEFF: float = 0.10                   # Lateral cone expansion factor
    MIN_CONE_RADIUS: float = 0.35                           # Minimum radius at t=0 (m)


@dataclass(frozen=True)
class PipelineConfig:
    """
    WebSocket telemetry server and runtime cloud deployment configuration.
    Resolves host and port dynamically from environment variables for container/cloud deployment
    (Railway, Render, Docker) while preserving 100% local default compatibility.
    """
    WS_HOST: str = field(default_factory=lambda: os.environ.get("WS_HOST", "0.0.0.0"))
    WS_PORT: int = field(default_factory=lambda: int(os.environ.get("PORT", os.environ.get("WS_PORT", "8765"))))
    TARGET_FPS: float = field(default_factory=lambda: float(os.environ.get("TARGET_FPS", "25.0")))
    MAX_PAYLOAD_CELLS: int = field(default_factory=lambda: int(os.environ.get("MAX_PAYLOAD_CELLS", "400")))
    PING_INTERVAL: float = field(default_factory=lambda: float(os.environ.get("WS_PING_INTERVAL", "20.0")))
    PING_TIMEOUT: float = field(default_factory=lambda: float(os.environ.get("WS_PING_TIMEOUT", "20.0")))
    ALLOWED_ORIGINS: str = field(default_factory=lambda: os.environ.get("ALLOWED_ORIGINS", "*"))

    @property
    def HOST(self) -> str:
        return self.WS_HOST

    @property
    def PORT(self) -> int:
        return self.WS_PORT


ServerConfig = PipelineConfig


@dataclass(frozen=True)
class Nav2Config:
    """ROS 2 nav_msgs/msg/OccupancyGrid planar costmap configurations."""
    RESOLUTION: float = 0.10            # 10 cm per grid pixel (or 0.05m)
    WIDTH: int = 400                    # 40m span / 0.10m = 400 cells
    HEIGHT: int = 400                   # 40m span / 0.10m = 400 cells
    ORIGIN_X: float = -20.0             # World frame X min (meters)
    ORIGIN_Y: float = -20.0             # World frame Y min (meters)
    FRAME_ID: str = "map"

    # ROS 2 Standard Cost Values
    COST_FREE: int = 0
    COST_CAUTION: int = 50
    COST_LETHAL: int = 100              # 100 in ROS 2 OccupancyGrid denotes lethal obstacle
    COST_UNKNOWN: int = -1              # -1 denotes unobserved cell


@dataclass(frozen=True)
class DatabaseConfig:
    """Redis state cache and telemetry configuration."""
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    SOCKET_TIMEOUT: float = 0.05        # 50ms non-blocking socket timeout
    KEY_POSE: str = "vehicle:pose"
    KEY_TRACKS: str = "tracks:active"
    KEY_TELEMETRY: str = "telemetry:stream"


# Global instances
BOUNDS = SpatialBounds()
QUADTREE = QuadtreeConfig()
COSTMAP = CostmapConfig()
GROUND_SEG = GroundSegmentationConfig()
TRACKING = TrackingConfig()
ROLLOUT = RolloutConfig()
CONFIG = PipelineConfig()
SERVER = CONFIG
NAV2 = Nav2Config()
DATABASE = DatabaseConfig()

