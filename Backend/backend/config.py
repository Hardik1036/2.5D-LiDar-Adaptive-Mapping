"""
Global configuration constants and parameters for SIH 2026 Problem Statement 53:
DRDO: Adaptive Variable Resolution 2.5D LiDAR Mapping for Dynamic Perception.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

# Set active dataset directory to dynamic KITTI dataset with robust fallback resolution
BACKEND_DIR = Path(__file__).resolve().parent.parent

def _resolve_dataset_path(*relative_paths: str) -> Path:
    for rel in relative_paths:
        for candidate in [
            BACKEND_DIR / rel,
            BACKEND_DIR / "data" / rel,
            Path(rel),
            BACKEND_DIR.parent / rel,
            Path("/app") / rel,
            Path("/app/Backend") / rel,
            Path("/app/data") / rel,
        ]:
            if candidate.exists() and (candidate.is_dir() or any(candidate.glob("*.bin"))):
                return candidate
    return BACKEND_DIR / "data" / relative_paths[0]

KITTI_DYNAMIC_DIR = _resolve_dataset_path(
    "data/dynamic_corridor/velodyne",
    "dynamic_corridor/velodyne",
    "data/sweeps",
)
KITTI_CLEAN_DIR = _resolve_dataset_path(
    "data/static_corridor/velodyne",
    "static_corridor/velodyne",
    "data/sweeps",
)

DEFAULT_DATASET_DIR = KITTI_CLEAN_DIR if (KITTI_CLEAN_DIR.exists() and any(KITTI_CLEAN_DIR.glob("*.bin"))) else KITTI_DYNAMIC_DIR
DATASET_DIR = DEFAULT_DATASET_DIR



@dataclass
class MapBounds:
    """Operational 3D bounding box for LiDAR perception (meters)."""
    X_MIN: float = -20.0   # 20 meters behind sensor
    X_MAX: float = 50.0    # 50 meters forward look-ahead
    Y_MIN: float = -15.0   # 15 meters left
    Y_MAX: float = 15.0    # 15 meters right
    Z_MIN: float = -2.50   # Ditch / drop-off floor
    Z_MAX: float = 2.0     # Ceiling clearance

    @property
    def x_range(self) -> float:
        return self.X_MAX - self.X_MIN

    @property
    def y_range(self) -> float:
        return self.Y_MAX - self.Y_MIN

    @property
    def z_range(self) -> float:
        return self.Z_MAX - self.Z_MIN

    def __getitem__(self, item: str) -> float:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def keys(self) -> List[str]:
        return ["X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"]

    def values(self) -> List[float]:
        return [getattr(self, k) for k in self.keys()]

    def items(self) -> List[Tuple[str, float]]:
        return [(k, getattr(self, k)) for k in self.keys()]

    def __iter__(self):
        return iter(self.keys())

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.keys()}


SpatialBounds = MapBounds


@dataclass(frozen=True)
class QuadtreeConfig:
    """Adaptive 2.5D Quadtree spatial partitioning configuration."""
    COARSE_RESOLUTION: float = 0.50  # 50 cm coarse cell size
    FINE_RESOLUTION: float = 0.05    # 5 cm fine cell size (leaf minimum)
    
    # Adaptive subdivision criteria:
    # Subdivide if internal point height variance > TAU_SIGMA or delta_z > TAU_Z
    TAU_SIGMA: float = 0.08          # Variance threshold (m^2)
    TAU_Z: float = 0.22              # Step height / elevation delta threshold (m, chassis clearance limit)
    
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
    MAX_TRAVERSABLE_SLOPE_DEG: float = 20.0  # Safe slope angle limit (degrees)
    LETHAL_SLOPE_DEG: float = 38.0           # Lethal rollover slope threshold (degrees)
    
    # Obstacle step height threshold in meters
    MAX_STEP_HEIGHT: float = 0.15            # Traversable flat ground step height limit (15cm)
    LETHAL_STEP_HEIGHT: float = 0.25         # Non-traversable lethal obstacle step (m)

    # Absolute drivable ground baseline relative to LiDAR sensor mount (m)
    # KITTI/Velodyne sensor mount sits at Z ≈ 1.73m above ground.
    GROUND_MIN: float = -2.15                # Lowest drivable ground level (m)
    GROUND_MAX: float = -1.35                # Highest drivable ground level (m)
    ELEVATED_OBSTACLE_Z: float = -1.2        # Elevated objects in air / rigid obstacles (m)
    SAFE_STEP_HEIGHT: float = 0.15           # Safe flat step height limit (15cm)
    CAUTION_STEP_HEIGHT: float = 0.25        # Caution curb / transition limit (m)


@dataclass(frozen=True)
class GroundSegmentationConfig:
    """Ground vs non-ground point cloud segmentation parameters."""
    NUM_RADIAL_BINS: int = 16
    NUM_ANGULAR_SECTORS: int = 32
    MAX_GROUND_SLOPE_DEG: float = 12.0
    SENSOR_HEIGHT_NOMINAL: float = 1.6  # Typical sensor mount height above ground (m)
    GROUND_DISTANCE_THRESHOLD: float = 0.18  # Distance to estimated ground plane (m, accommodates < 15cm roughness)
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
    MAX_TRACKS: int = 40                # Maximum tracked objects to prevent explosion


@dataclass(frozen=True)
class RolloutConfig:
    """Trajectory rollout and expanding hazard cone settings."""
    HORIZONS: Tuple[float, float, float] = (1.0, 2.0, 3.0)  # Projection horizons (seconds)
    VELOCITY_UNCERTAINTY_COEFF: float = 0.12                # Forward cone expansion factor
    LATERAL_EXPANSION_COEFF: float = 0.08                   # Lateral cone expansion factor
    MIN_CONE_RADIUS: float = 0.25                           # Minimum radius at t=0 (m)


@dataclass(frozen=True)
class PipelineConfig:
    """
    WebSocket telemetry server and runtime cloud deployment configuration.
    Resolves host and port dynamically from environment variables for container/cloud deployment
    (Railway, Render, Docker) while preserving 100% local default compatibility.
    """
    WS_HOST: str = field(default_factory=lambda: os.environ.get("WS_HOST", "0.0.0.0"))
    WS_PORT: int = field(default_factory=lambda: int(os.environ.get("PORT", os.environ.get("WS_PORT", "8765"))))
    INGESTION_HZ: float = field(default_factory=lambda: float(os.environ.get("INGESTION_HZ", "20.0")))
    TARGET_FPS: float = field(default_factory=lambda: float(os.environ.get("TARGET_FPS", "20.0")))
    MAX_PAYLOAD_CELLS: int = field(default_factory=lambda: int(os.environ.get("MAX_PAYLOAD_CELLS", "1500")))
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


@dataclass(frozen=True)
class PointPillarsConfig:
    """
    PointPillars (Model 2) 3D Object Detection configuration.
    Extracted from backend/models/configs/cbgs_pp_multihead.yaml.
    """
    POINT_CLOUD_RANGE: Tuple[float, float, float, float, float, float] = (
        -51.2, -51.2, -5.0, 51.2, 51.2, 3.0
    )
    VOXEL_SIZE: Tuple[float, float, float] = (0.2, 0.2, 8.0)
    CLASS_NAMES: Tuple[str, ...] = (
        "car",
        "truck",
        "construction_vehicle",
        "bus",
        "trailer",
        "barrier",
        "motorcycle",
        "bicycle",
        "pedestrian",
        "traffic_cone",
    )
    CODE_SIZE: int = 9
    MAX_POINTS_PER_VOXEL: int = 20
    MAX_NUMBER_OF_VOXELS: int = 30000

    @property
    def x_min(self) -> float:
        return self.POINT_CLOUD_RANGE[0]

    @property
    def y_min(self) -> float:
        return self.POINT_CLOUD_RANGE[1]

    @property
    def z_min(self) -> float:
        return self.POINT_CLOUD_RANGE[2]

    @property
    def x_max(self) -> float:
        return self.POINT_CLOUD_RANGE[3]

    @property
    def y_max(self) -> float:
        return self.POINT_CLOUD_RANGE[4]

    @property
    def z_max(self) -> float:
        return self.POINT_CLOUD_RANGE[5]

    @property
    def grid_size(self) -> Tuple[int, int, int]:
        """Calculates BEV grid dimensions (W, H, D) based on point cloud range and voxel size."""
        dx = int(round((self.x_max - self.x_min) / self.VOXEL_SIZE[0]))
        dy = int(round((self.y_max - self.y_min) / self.VOXEL_SIZE[1]))
        dz = int(round((self.z_max - self.z_min) / self.VOXEL_SIZE[2]))
        return (dx, dy, dz)

    @classmethod
    def from_yaml(cls, yaml_path: Optional[str] = None) -> "PointPillarsConfig":
        """
        Loads configuration dynamically from cbgs_pp_multihead.yaml if available,
        falling back to default constants.
        """
        if yaml_path is None:
            from pathlib import Path
            root = Path(__file__).resolve().parent
            candidates = [
                root / "models" / "configs" / "cbgs_pp_multihead.yaml",
                Path("backend/models/configs/cbgs_pp_multihead.yaml"),
                Path("models/configs/cbgs_pp_multihead.yaml"),
            ]
            for c in candidates:
                if c.exists():
                    yaml_path = str(c)
                    break

        if yaml_path:
            try:
                import yaml
                with open(yaml_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                class_names = tuple(data.get("CLASS_NAMES", cls.CLASS_NAMES))
                data_cfg = data.get("DATA_CONFIG", {})
                pc_range = tuple(float(x) for x in data_cfg.get("POINT_CLOUD_RANGE", cls.POINT_CLOUD_RANGE))
                voxel_size = cls.VOXEL_SIZE
                for proc in data_cfg.get("DATA_PROCESSOR", []):
                    if "VOXEL_SIZE" in proc:
                        voxel_size = tuple(float(v) for v in proc["VOXEL_SIZE"])
                        break
                return cls(
                    POINT_CLOUD_RANGE=pc_range,
                    VOXEL_SIZE=voxel_size,
                    CLASS_NAMES=class_names,
                )
            except Exception:
                pass
        return cls()


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
POINT_PILLARS = PointPillarsConfig.from_yaml()

# Core algorithmic and spatial configuration aliases
MAP_BOUNDS = BOUNDS
SPATIAL_BOUNDS = BOUNDS
QUADTREE_CONFIG = QUADTREE
COSTMAP_CONFIG = COSTMAP
TRACKING_CONFIG = TRACKING
WEBSOCKET_CONFIG = CONFIG




