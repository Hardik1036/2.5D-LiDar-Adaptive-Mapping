"""
Pipeline Orchestrator for SIH 2026 Problem Statement 53:
DRDO: Adaptive Variable Resolution 2.5D LiDAR Mapping for Dynamic Perception.

Real-time Tactical Loop:
  1. Ingest Point Cloud Buffer in memory ([N, 3] or [N, C] float32)
  2. Statistical Dust & Atmospheric Scatter Removal (< 2ms) [F1.3]
  3. Ground vs Non-Ground Plane Segmentation (< 4ms)
  4. Millimeter-Residual Thin Hazard / Spike Strip Detection (< 1ms) [F1.6]
  5. ML Semantic Classification & 3D Bounding Box Ingestion (< 1ms)
  6. Adaptive 2.5D Quadtree Construction (< 5ms)
  7. Obstacle Clustering & Porosity Analysis (< 5ms) [F1.5]
  8. 2D Constant Velocity Kalman Tracking (< 2ms)
  9. Negative Obstacle & Trench Drop-off Detection (< 1ms) [F5.1]
  10. Temporal Map Blending & 1D Elevation Kalman Memory (< 1ms) [F2.4]
  11. Ghost Clearing & Trajectory Rollout Cones (< 2ms)
  12. Traversability Costmap Evaluation & Vegetation Scaling (< 2ms)
  13. ROS 2 nav_msgs/OccupancyGrid Generation (< 3ms)
  14. State Database Sync & Async WebSocket Broadcast (ws://localhost:8765)
Target: >= 25 Hz throughput (< 35 ms frame latency).
"""

import os
from pathlib import Path

# Load project-level .env if present
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

import argparse
import asyncio
import gc
import json
import logging
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Union

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None

import numpy as np

from backend.adapters.db_adapter import DatabaseAdapter
from backend.adapters.ml_adapter import MLPerceptionAdapter
from backend.config import (
    BOUNDS,
    CONFIG,
    COSTMAP,
    DATABASE,
    DEFAULT_DATASET_DIR,
    NAV2,
    QUADTREE,
    SERVER,
    TRACKING,
    SPATIAL_BOUNDS,
    QUADTREE_CONFIG,
    COSTMAP_CONFIG,
    TRACKING_CONFIG,
    WEBSOCKET_CONFIG,
)
from backend.ingestion.dust_filter import StatisticalDustFilter
from backend.ingestion.ground_segmentation import GroundSegmenter
from backend.ingestion.thin_hazard_detector import ThinHazardDetector
from backend.ingestion.dataset_loader import DatasetLoader, resolve_sweep_directory
from backend.mapping.costmap import CostmapEvaluator
from backend.mapping.degraded_mode import SensorHealthMonitor
from backend.mapping.negative_obstacles import TrenchDetector
from backend.mapping.occupancy_publisher import OccupancyGridBuilder
from backend.mapping.porosity_filter import PorosityClassifier
from backend.mapping.quadtree import AdaptiveQuadtree
from backend.mapping.temporal_blender import TemporalMapBlender
from backend.mapping.vegetation_filter import VegetationFilter
from backend.mapping.memory_benchmark import (
    compute_theoretical_reduction,
    measure_process_rss_mib,
    measure_spatial_index_mapping_memory_mib,
    run_empirical_memory_benchmark,
)
from backend.server.payload_builder import PayloadBuilder
from backend.server.websocket_server import TelemetryWebSocketServer
from backend.telemetry.telemetry_db import AsyncTelemetryDB
from backend.tracking.clustering import EuclideanClusterer
from backend.tracking.ghost_clearing import GhostClearing
from backend.tracking.kalman_tracker import KalmanTracker
from backend.tracking.object_detector import ObjectDetector3D, exclude_dynamic_points_from_elevation
from backend.tracking.trajectory_rollout import TrajectoryRollout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PerceptionPipeline")


def _generate_live_sensor_buffer(frame_idx: int, dt: float = 0.05) -> np.ndarray:
    """
    Generates an in-memory tactical LiDAR point buffer for live frontend streaming.
    Zero disk access: generates realistic asphalt ground, dynamic vehicles, and obstacles in RAM.
    """
    rng = np.random.RandomState(42 + (frame_idx % 500))

    # Road ground surface
    n_road = 6000
    rx = rng.uniform(-15.0, 35.0, n_road).astype(np.float32)
    ry = rng.uniform(-4.0, 4.0, n_road).astype(np.float32)
    rz = -1.60 + rng.normal(0.0, 0.005, n_road).astype(np.float32)
    ri = rng.uniform(0.15, 0.35, n_road).astype(np.float32)

    # Curbs & Sidewalks
    n_sw = 1500
    sw_x = rng.uniform(-15.0, 35.0, n_sw).astype(np.float32)
    sw_side = rng.choice([-1.0, 1.0], n_sw).astype(np.float32)
    sw_y = sw_side * rng.uniform(4.0, 7.5, n_sw).astype(np.float32)
    sw_z = -1.50 + rng.normal(0.0, 0.006, n_sw).astype(np.float32)
    sw_i = rng.uniform(0.20, 0.40, n_sw).astype(np.float32)

    # Dynamic Moving Vehicle cruising forward along x
    t = (frame_idx % 200) * dt
    veh_center_x = float(2.0 + 4.0 * t)
    veh_center_y = float(1.8)
    veh_center_z = float(-0.9)

    n_veh = 800
    vl, vw, vh = 4.2, 1.8, 1.4
    vx = rng.uniform(veh_center_x - vl * 0.5, veh_center_x + vl * 0.5, n_veh).astype(np.float32)
    vy = rng.uniform(veh_center_y - vw * 0.5, veh_center_y + vw * 0.5, n_veh).astype(np.float32)
    vz = rng.uniform(veh_center_z - vh * 0.5, veh_center_z + vh * 0.5, n_veh).astype(np.float32)
    vi = rng.uniform(0.70, 0.90, n_veh).astype(np.float32)

    # Static Obstacles
    n_obs = 300
    ox = rng.normal(14.0, 0.1, n_obs).astype(np.float32)
    oy = rng.normal(-5.2, 0.1, n_obs).astype(np.float32)
    oz = rng.uniform(-1.5, 1.8, n_obs).astype(np.float32)
    oi = rng.uniform(0.4, 0.6, n_obs).astype(np.float32)

    all_x = np.concatenate([rx, sw_x, vx, ox])
    all_y = np.concatenate([ry, sw_y, vy, oy])
    all_z = np.concatenate([rz, sw_z, vz, oz])
    all_i = np.concatenate([ri, sw_i, vi, oi])
    total_pts = len(all_x)
    ring = (np.arange(total_pts) % 64).astype(np.float32)

    return np.column_stack([all_x, all_y, all_z, all_i, ring]).astype(np.float32)


class PerceptionPipeline:
    """
    Coordinates the full real-time tactical LiDAR perception stack.
    Accepts live sensor point cloud arrays directly in memory via process_frame().
    """

    def __init__(
        self,
        host: str = SERVER.HOST,
        port: int = SERVER.PORT,
        target_fps: float = SERVER.TARGET_FPS,
        use_redis: bool = False,
        ros_output: bool = False,
        profile_mode: bool = False,
        dataset_dir: Optional[Union[str, Path]] = None,
        kaggle: bool = False,
        kaggle_dataset: Optional[str] = None,
        kaggle_streamer: Optional[Any] = None,
        **kwargs,
    ):
        self.config = CONFIG
        self.target_fps = target_fps
        self.dt = 1.0 / target_fps
        self.is_running = False
        self.ros_output = ros_output
        self.profile_mode = profile_mode
        self.kaggle = False
        self.kaggle_dataset = None
        self.kaggle_streamer = None
        ds_path = Path(dataset_dir) if dataset_dir is not None else DEFAULT_DATASET_DIR
        # Ensure path points strictly to the full sweeps in training/velodyne, never parent or gt_database
        if ds_path.is_dir():
            velo_sub = ds_path / "training" / "velodyne"
            if velo_sub.exists():
                ds_path = velo_sub
            elif (ds_path / "velodyne").exists():
                ds_path = ds_path / "velodyne"
        self.dataset_dir = ds_path

        logger.info(f"Perception pipeline initialized for live memory ingestion. Target: {self.target_fps} Hz")

        # Tactical Ingestion Preprocessors & Hazard Detectors
        self.dust_filter = StatisticalDustFilter()
        self.ground_seg = GroundSegmenter()
        self.thin_hazard_detector = ThinHazardDetector()

        # ML Integration & Database Adapters
        self.ml_adapter = MLPerceptionAdapter()
        self.db_adapter = DatabaseAdapter(enabled=use_redis)

        # 2.5D Adaptive Quadtree Mapping, Terrain Memory, & Tactical Evaluators
        self.quadtree = AdaptiveQuadtree()
        self.bounds = self.quadtree.bounds
        self.porosity_classifier = PorosityClassifier()
        self.trench_detector = TrenchDetector()
        self.temporal_blender = TemporalMapBlender()
        self.costmap = CostmapEvaluator()
        self.vegetation_filter = VegetationFilter()
        self.sensor_health = SensorHealthMonitor()
        self.occupancy_builder = OccupancyGridBuilder() if ros_output else None

        # Dynamic Perception & Tracking
        self.clusterer = EuclideanClusterer()
        self.tracker = KalmanTracker()
        self.ghost_clearing = GhostClearing()
        self.rollout = TrajectoryRollout()

        # Server & Streaming
        self.server = TelemetryWebSocketServer(host=host, port=port)
        self.payload_builder = PayloadBuilder(max_cells=10000, max_raw_points=5000, round_decimals=2)
        self.telemetry_db = AsyncTelemetryDB()

        # Dataset loader & dynamic sequence routing
        self.current_dataset_mode = "static"
        if dataset_dir is not None:
            self.dataset_loader = DatasetLoader(mode_or_dir=self.dataset_dir, mode=self.current_dataset_mode)
        else:
            self.dataset_loader = DatasetLoader(mode=self.current_dataset_mode)
        self.loader = self.dataset_loader
        self.dataset_dir = self.dataset_loader.data_dir

        # Dynamic Playback Controls (HUD WebSocket listener)
        self.is_paused = False
        self._step_once = False
        self.playback_speed = 1.0
        self.server.set_command_callback(self.handle_client_command)
        self.server.pipeline = self
        self.server.loader = self.loader

        # Metrics
        self.frame_count = 0
        self.rolling_fps = 0.0
        self.rolling_latency_ms = 0.0
        self._last_ram_mb: Optional[float] = None
        self._last_vram_mb: Optional[float] = None
        self._warmed_up = False

    def step_next_frame(self) -> None:
        """Advances single frame safely without unhandled exception."""
        self.is_paused = True
        if hasattr(self, "server") and self.server:
            self.server.is_paused = True
        self._step_once = True
        logger.info("[Perception] Stepping to next frame.")

    def switch_dataset(self, mode: str) -> bool:
        """
        Hot-swaps active dataset mode ('static' or 'dynamic') cleanly
        without setting file lists to empty or raising IndexError.
        """
        target_mode = "dynamic" if "dynamic" in str(mode).lower() else "static"
        desc = (
            "Clean Static Urban Road Corridor (Backend/data/static_corridor)"
            if target_mode == "static"
            else "Continuous Dynamic Multi-Object Tracking (Backend/data/dynamic_corridor)"
        )
        logger.info(f"[Perception] Switching dataset stream to: {desc} (mode: {target_mode})")

        swapped = False
        for attr_name in ['loader', 'dataset_loader', 'streamer', 'kaggle_streamer']:
            loader_instance = getattr(self, attr_name, None)
            if loader_instance is not None:
                if hasattr(loader_instance, 'switch_dataset'):
                    swapped = loader_instance.switch_dataset(target_mode)
                    if swapped:
                        self.dataset_dir = getattr(loader_instance, "data_dir", self.dataset_dir)
                        break
                elif hasattr(loader_instance, 'set_directory'):
                    target_dir = resolve_sweep_directory(preferred_mode=target_mode)
                    if target_dir and target_dir.exists():
                        swapped = loader_instance.set_directory(str(target_dir))
                        if swapped:
                            self.dataset_dir = target_dir
                            break

        self.current_dataset_mode = target_mode
        sweep_count = len(getattr(self.dataset_loader, "files", [])) if hasattr(self, "dataset_loader") and self.dataset_loader else 0
        if hasattr(self, 'server') and self.server:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self.server.broadcast(json.dumps({
                        "type": "dataset_swapped",
                        "action": "dataset_swapped",
                        "mode": target_mode,
                        "current_mode": target_mode,
                        "sweep_count": sweep_count,
                        "description": desc
                    }))
                )
            except RuntimeError:
                pass
        return swapped

    def seek_frame(self, target_frame: int) -> None:
        """Safely seeks to target frame index."""
        logger.info(f"[Perception] Seeking to frame {target_frame}")
        for attr_name in ['loader', 'dataset_loader']:
            loader_instance = getattr(self, attr_name, None)
            if loader_instance is not None:
                if hasattr(loader_instance, 'seek_frame'):
                    loader_instance.seek_frame(target_frame)
                elif hasattr(loader_instance, 'files') and loader_instance.files:
                    idx = max(0, min(int(target_frame), len(loader_instance.files) - 1))
                    loader_instance.current_index = idx
                    loader_instance.current_idx = idx

    def handle_client_command(self, cmd_data: dict) -> None:
        """
        Processes dynamic runtime control commands received from connected frontend HUD clients.
        Supported actions:
          - 'pause': Halts sweep progression without closing WebSocket or dropping client state
          - 'resume' / 'play': Resumes sweep loop execution
          - 'toggle_pause': Inverts current pause state
          - 'next' / 'step': Steps forward a single frame
          - 'set_dataset' / 'switch_dataset': Swaps active corridor mode
          - 'seek': Seeks to frame index
          - 'set_speed': Scales frame pacing interval (e.g. 0.25x, 0.5x, 1.0x, 2.0x)
        """
        if not isinstance(cmd_data, dict):
            return
        action = str(cmd_data.get("action") or cmd_data.get("command") or cmd_data.get("type") or "").lower().strip()
        if not action or action in ("ping", "pong"):
            return

        try:
            if action in ("pause", "playback_pause"):
                self.is_paused = True
                if hasattr(self, "server") and self.server:
                    self.server.is_paused = True
                logger.info("Pipeline playback PAUSED via client command.")
            elif action in ("resume", "play", "playback_resume", "playback_play"):
                self.is_paused = False
                if hasattr(self, "server") and self.server:
                    self.server.is_paused = False
                logger.info("Pipeline playback RESUMED via client command.")
            elif action in ("toggle_pause", "pause_toggle"):
                self.is_paused = not self.is_paused
                if hasattr(self, "server") and self.server:
                    self.server.is_paused = self.is_paused
                logger.info(f"Pipeline playback pause toggled: {self.is_paused}")
            elif action in ("next", "step", "step_forward", "playback_next"):
                self.step_next_frame()
            elif action in ("set_dataset", "switch_dataset", "dataset_toggle"):
                target_mode = str(cmd_data.get("mode") or cmd_data.get("dataset") or "static").lower()
                self.switch_dataset(target_mode)
            elif action == "seek":
                target_frame = int(cmd_data.get("frame", 0))
                self.seek_frame(target_frame)
            elif action == "set_speed":
                try:
                    speed = float(cmd_data.get("speed") or cmd_data.get("value", 1.0))
                    self.playback_speed = max(0.1, min(10.0, speed))
                    logger.info(f"Pipeline playback speed set to: {self.playback_speed:.2f}x")
                except (ValueError, TypeError):
                    pass
        except Exception as e:
            logger.error(f"[Perception] Error handling client command '{action}': {e}", exc_info=True)

    def warmup(self, frames: int = 2, sample_points: Optional[np.ndarray] = None) -> None:
        """Pre-warms pipeline buffers, JIT compilations, and model inference sessions in memory."""
        try:
            if sample_points is None:
                # In-memory synthetic ground plane grid: no disk access or file loading
                gx = np.linspace(BOUNDS.X_MIN / 2, BOUNDS.X_MAX / 2, 40)
                gy = np.linspace(BOUNDS.Y_MIN / 2, BOUNDS.Y_MAX / 2, 40)
                xx, yy = np.meshgrid(gx, gy)
                zz = np.full_like(xx, -1.60)  # Road baseline to warm ThreatNet1D & plane estimators
                # Synthetic vehicle obstacle box to fully pre-warm clustering, porosity, and tracking
                obs_box = (np.abs(xx - 6.0) < 2.0) & (np.abs(yy) < 1.0)
                zz[obs_box] = 0.20
                sample_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)

            raw_points = sample_points.copy()
            mask = (
                (raw_points[:, 0] >= BOUNDS.X_MIN) & (raw_points[:, 0] <= BOUNDS.X_MAX) &
                (raw_points[:, 1] >= BOUNDS.Y_MIN) & (raw_points[:, 1] <= BOUNDS.Y_MAX) &
                (raw_points[:, 2] >= BOUNDS.Z_MIN) & (raw_points[:, 2] <= BOUNDS.Z_MAX)
            )
            raw_points = raw_points[mask]

            isolated_blender = TemporalMapBlender()

            for _ in range(frames):
                clean_points = self.dust_filter.filter(raw_points.copy())
                ground_pts, obstacle_pts, _ = self.ground_seg.segment(clean_points)
                th = self.thin_hazard_detector.detect_threats(ground_pts)
                self.ml_adapter.segment_point_cloud(clean_points)
                leaves = self.quadtree.build(clean_points)
                leaves = self.quadtree.interpolate_ground_rings()
                leaves = self.quadtree.apply_clearance_coarsening()
                if th:
                    self.thin_hazard_detector.apply_hazards_to_leaves(leaves, th)
                cl = self.clusterer.cluster(obstacle_pts)
                p_cl = self.porosity_classifier.classify_clusters(cl)
                self.costmap.evaluate_leaves(leaves)
                isolated_blender.blend_quadtree(leaves)

            self.temporal_blender.clear()
            self.tracker.tracks.clear()
            self.frame_count = 0
            self.rolling_fps = 0.0
            self.rolling_latency_ms = 0.0
            gc.collect()
            self._warmed_up = True
        except Exception as e:
            logger.debug(f"Pipeline warmup exception: {e}")

    def process_frame(
        self,
        points: np.ndarray,
        timestamp: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
        frame_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Primary entry point for live point cloud ingestion.
        Processes point arrays passed directly as input arguments in memory.

        Args:
            points: [N, 3], [N, 4], or [N, 5] numpy float array of LiDAR coordinates.
            timestamp: Sensor acquisition timestamp in seconds (defaults to current time).
            meta: Optional metadata dictionary (e.g. semantic labels, dynamic detections).
            frame_id: Optional frame index for tracking and telemetry logging.

        Returns:
            Dictionary containing processed frame results, leaves, tracks, and telemetry.
        """
        t0 = time.perf_counter()
        if timestamp is None:
            timestamp = time.time()
        if meta is None:
            meta = {}
        if frame_id is not None:
            self.frame_idx = frame_id
            self.frame_count = frame_id
        else:
            self.frame_count += 1

        # Ego-motion clearing: completely reset local quadtree and temporal terrain memory
        # to prevent stationary roadside objects from smearing/streaking in sensor frame
        self.quadtree.clear()
        self.temporal_blender.clear()

        # 1. Ingestion Validation & Operational 50-meter driving envelope
        t_ingest_start = time.perf_counter()
        raw_points = np.asarray(points, dtype=np.float32)
        if raw_points.ndim == 1:
            raw_points = raw_points.reshape(-1, 3 if raw_points.shape[0] % 3 == 0 else 4)

        # Operational 50-meter driving envelope:
        x = raw_points[:, 0]
        y = raw_points[:, 1]
        z = raw_points[:, 2]

        valid_mask = (
            (x >= BOUNDS.X_MIN) & (x <= BOUNDS.X_MAX) &
            (y >= BOUNDS.Y_MIN) & (y <= BOUNDS.Y_MAX) &
            (z >= BOUNDS.Z_MIN) & (z <= BOUNDS.Z_MAX) &
            np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        )
        clean_points = raw_points[valid_mask]
        # Fast vectorized downsample for ultra-dense sweeps (> 65,000 points) to cap cycle under 120ms
        if len(clean_points) > 65000:
            clean_points = clean_points[::2]
        t_ingest = (time.perf_counter() - t_ingest_start) * 1000.0

        # 2. Statistical Dust Outlier Filtering [F1.3]
        t_dust_start = time.perf_counter()
        clean_points = self.dust_filter.filter(clean_points)
        t_dust = (time.perf_counter() - t_dust_start) * 1000.0

        # Sensor health & FOV quadrant density monitor [F5.3]
        health_info = self.sensor_health.update(clean_points)

        # 3. Ground plane segmentation
        t_seg_start = time.perf_counter()
        ground_pts, obstacle_pts, ground_mask = self.ground_seg.segment(clean_points)
        t_seg = (time.perf_counter() - t_seg_start) * 1000.0

        # 4. Thin Hazard & Spike Strip Detection [F1.6] (ThreatNet1D ONNX)
        t_thin_start = time.perf_counter()
        thin_hazards = self.thin_hazard_detector.detect_threats(ground_pts)
        t_thin = (time.perf_counter() - t_thin_start) * 1000.0

        # 5. ML Perception Adapter (Model 1 SalsaNext Semantic Segmentation & Model 2 ThreatNet1D)
        t_ml_start = time.perf_counter()
        semantic_labels = meta.get("semantic_labels") if isinstance(meta, dict) else None
        if semantic_labels is None:
            semantic_labels = self.ml_adapter.segment_point_cloud(clean_points)
        ml_subsets = self.ml_adapter.process_semantic_labels(clean_points, semantic_labels)
        pillar_dets = self.ml_adapter.process_pillar_detections(
            meta.get("dynamic_detections") if isinstance(meta, dict) else None
        )

        # Low-profile threat detection via Model 2 (ThreatNet1D) on isolated semantic ground
        ground_points = clean_points[semantic_labels == 0]
        ground_plane = getattr(self.thin_hazard_detector, "_last_plane", None)
        if ground_plane is None and len(ground_points) >= 3:
            ground_plane = self.thin_hazard_detector.fit_ground_plane(ground_points)
        threat_mask = self.ml_adapter.detect_low_profile_threats(ground_points, ground_plane)
        threat_points = ground_points[threat_mask]
        t_ml = (time.perf_counter() - t_ml_start) * 1000.0

        # Dynamic Point Exclusion (PS 26053 Step 4.3): exclude confirmed dynamic tracks before cell stats
        confirmed_dyn = self.tracker.get_dynamic_tracks()
        elevation_points = exclude_dynamic_points_from_elevation(clean_points, confirmed_dyn)

        # 6. Adaptive 2.5D Quadtree construction
        t_quad_start = time.perf_counter()
        point_costs = np.zeros(len(elevation_points), dtype=np.int32)
        if len(threat_points) > 0 and len(elevation_points) == len(clean_points):
            ground_indices = np.nonzero(semantic_labels == 0)[0]
            point_costs[ground_indices[threat_mask]] = 255

        leaves = self.quadtree.build(elevation_points, point_costs=point_costs)
        leaves = self.quadtree.interpolate_ground_rings()
        # leaves = self.quadtree.interpolate_obstacle_clusters()  # Bypassed: map only real physical obstacle returns
        leaves = self.quadtree.apply_clearance_coarsening()
        # Mark detected spike/hazard coordinates directly into 2.5D Quadtree leaf costs (cost = 255)
        if thin_hazards:
            self.thin_hazard_detector.apply_hazards_to_leaves(leaves, thin_hazards)
        t_quad = (time.perf_counter() - t_quad_start) * 1000.0

        # 7. Obstacle clustering & Porosity Classification [F1.5]
        t_track_start = time.perf_counter()
        if len(obstacle_pts) > 3000:
            clustering_pts = obstacle_pts[::3]
        else:
            clustering_pts = obstacle_pts
        clusters = self.clusterer.cluster(clustering_pts)
        classified_clusters = self.porosity_classifier.classify_clusters(clusters)

        # Extract stationary parked vehicles from raw cluster geometries (zero-velocity, confirmed dimensions)
        parked_car_clusters = [c for (c, is_porous, _) in classified_clusters if c.label == "parked_car"]

        # Filter out soft porous vegetation clusters from rigid dynamic tracker
        rigid_candidates = [c for (c, is_porous, _) in classified_clusters if not is_porous]
        all_candidates = rigid_candidates + pillar_dets
        active_tracks = self.tracker.update(all_candidates, dt=self.dt)
        dynamic_tracks = self.tracker.get_dynamic_tracks()
        t_track = (time.perf_counter() - t_track_start) * 1000.0

        # 8. Negative Obstacle / Trench Drop-off Detection [F5.1]
        t_trench_start = time.perf_counter()
        dropoffs = self.trench_detector.find_dropoffs(ground_pts)
        self.trench_detector.apply_dropoffs_to_leaves(leaves, dropoffs)
        t_trench = (time.perf_counter() - t_trench_start) * 1000.0

        # 9. Temporal Map Blending & 1D Kalman Terrain Memory [F2.4]
        t_blend_start = time.perf_counter()
        self.temporal_blender.blend_quadtree(leaves)
        t_blend = (time.perf_counter() - t_blend_start) * 1000.0

        # 10. Traversability costmap evaluation & vegetation scaling
        t_cost_start = time.perf_counter()
        self.costmap.evaluate_leaves(leaves)
        self.costmap.apply_obstacle_occupancy(leaves, obstacle_pts, dynamic_tracks=dynamic_tracks)

        # Apply vegetation filtering
        if len(ml_subsets["vegetation"]) > 0:
            self.vegetation_filter.apply_vegetation_scaling(
                leaves,
                vegetation_points=ml_subsets["vegetation"],
                rigid_points=ml_subsets["rigid"],
            )

        # Dynamic predicted hazard cones
        rollout_hazards = self.rollout.predict_hazards(active_tracks, max_hazards=5)
        if rollout_hazards:
            self.costmap.apply_dynamic_hazards(leaves, rollout_hazards)

        # 11. Ghost smear clearing & trajectory rollout (cleans road clutter and dynamic trails)
        t_roll_start = time.perf_counter()
        self.ghost_clearing.clear_ghosts(leaves, clean_points, dynamic_tracks=dynamic_tracks)
        self.ghost_clearing.register_dynamic_footprints(dynamic_tracks)
        hazard_cones = self.rollout.rollout_all(active_tracks)
        t_roll = (time.perf_counter() - t_roll_start) * 1000.0

        # 12. Degraded-Mode Safe Fallback [F5.3]
        if health_info["is_degraded"]:
            self.sensor_health.apply_degraded_costmap_inflation(leaves, caution_cost=180)
            self.sensor_health.preserve_terrain_memory(leaves, self.temporal_blender)

        t_cost = (time.perf_counter() - t_cost_start) * 1000.0

        # 13. ROS 2 nav_msgs/OccupancyGrid Generation (optional)
        t_ros_start = time.perf_counter()
        ros_occupancy_grid = None
        if self.ros_output and self.occupancy_builder is not None:
            grid_matrix = self.occupancy_builder.build_grid(leaves, rollout_hazards)
            ros_occupancy_grid = self.occupancy_builder.to_ros_message_dict(grid_matrix, timestamp=timestamp)
        t_ros = (time.perf_counter() - t_ros_start) * 1000.0

        # 14. State Database Synchronization
        t_db_start = time.perf_counter()
        if self.db_adapter.enabled:
            self.db_adapter.sync_active_tracks(active_tracks)
            self.db_adapter.log_frame_telemetry({
                "frame_id": self.frame_count,
                "fps": self.rolling_fps,
                "latency_ms": self.rolling_latency_ms,
                "cell_count": len(leaves),
                "system_status": health_info["system_status"],
            })
        t_db = (time.perf_counter() - t_db_start) * 1000.0

        # Compute cycle timings
        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        instant_fps = 1000.0 / max(total_latency_ms, 0.001)

        alpha = 0.2  # Exponential moving average factor for stats
        if self.rolling_fps == 0.0:
            self.rolling_fps = instant_fps
            self.rolling_latency_ms = total_latency_ms
        else:
            self.rolling_fps = (1 - alpha) * self.rolling_fps + alpha * instant_fps
            self.rolling_latency_ms = (1 - alpha) * self.rolling_latency_ms + alpha * total_latency_ms

        tree_stats = self.quadtree.get_statistics()
        safe_count = 0
        caution_count = 0
        hazard_count = 0
        for l in leaves:
            c = l.cost
            if c <= 50:
                safe_count += 1
            elif c <= 180:
                caution_count += 1
            else:
                hazard_count += 1

        quadtree_ram_mb = tree_stats.get("ram_mb", self.quadtree.get_memory_footprint_mb())
        ground_inlier_ratio = round(len(ground_pts) / max(len(clean_points), 1), 3)

        # Monotonic latency stages (Section 8.7)
        input_ingestion_ms = round(t_ingest, 2)
        preprocessing_ms = round(t_dust + t_seg, 2)
        semantic_segmentation_inference_ms = round(t_ml, 2) if self.ml_adapter.salsa_session is not None else 0.0
        mapping_ms = round(t_quad + t_trench + t_blend + t_cost, 2)
        three_d_detection_ms = round(t_thin, 2) # Including ThinHazard as 3D detection for now, + pointpillars later
        tracking_ms = round(t_track + t_roll, 2)
        output_construction_ms = round(t_ros + t_db, 2)
        
        sum_measured = t_ingest + t_dust + t_seg + t_ml + t_quad + t_trench + t_blend + t_cost + t_track + t_roll + t_ros + t_db + t_thin
        other_pipeline_overhead_ms = max(0.0, round(total_latency_ms - sum_measured, 2))

        # Memory accounting separation (Section 8.3 - 8.6)
        mapping_memory_mib = measure_spatial_index_mapping_memory_mib(self.quadtree)
        process_rss_mib = measure_process_rss_mib()
        uni_theo_mib, adap_theo_mib, theo_red_pct = compute_theoretical_reduction(BOUNDS, len(leaves))

        # Runtime mode determination (Section 1.3)
        if getattr(self, "_is_simulation", False) or (meta and meta.get("is_simulation", False)):
            runtime_mode = "SIMULATION"
        elif self.ml_adapter.salsa_session is not None:
            runtime_mode = "LIVE_DL"
        else:
            runtime_mode = "LIVE_GEOMETRIC_FALLBACK"

        # Empirical Uniform-vs-Adaptive benchmark on identical sweep (PS 26053 Item 2)
        if len(clean_points) > 0:
            emp_res = run_empirical_memory_benchmark(clean_points, bounds=BOUNDS, warm_up=False, repetitions=1)
            uniform_measured_mib = emp_res.uniform_5cm_measured_mapping_memory_mib
            adaptive_measured_mib = emp_res.adaptive_measured_mapping_memory_mib
            emp_reduction_pct = emp_res.measured_reduction_percent
        else:
            uniform_measured_mib = None
            adaptive_measured_mib = None
            emp_reduction_pct = None

        model_status_dict = self.ml_adapter.get_model_status()

        system_stats: Dict[str, Any] = {
            # Targets vs Measured
            "target_fps": 33,
            "target_latency_ms": 30,
            "actual_fps": round(self.rolling_fps, 1),
            "total_latency_ms": round(self.rolling_latency_ms, 2),
            "fps": round(self.rolling_fps, 1),
            "latency_ms": round(self.rolling_latency_ms, 2),

            # Latency stages
            "input_ingestion_ms": input_ingestion_ms,
            "preprocessing_ms": preprocessing_ms,
            "semantic_segmentation_inference_ms": semantic_segmentation_inference_ms,
            "inference_ms": semantic_segmentation_inference_ms,  # alias for backwards compatibility
            "mapping_ms": mapping_ms,
            "3d_detection_ms": three_d_detection_ms,
            "tracking_ms": tracking_ms,
            "output_construction_ms": output_construction_ms,
            "other_pipeline_overhead_ms": other_pipeline_overhead_ms,
            
            "profiling_breakdown": {
                "input_ingestion_ms": input_ingestion_ms,
                "preprocessing_ms": preprocessing_ms,
                "semantic_segmentation_inference_ms": semantic_segmentation_inference_ms,
                "mapping_ms": mapping_ms,
                "3d_detection_ms": three_d_detection_ms,
                "tracking_ms": tracking_ms,
                "output_construction_ms": output_construction_ms,
                "other_pipeline_overhead_ms": other_pipeline_overhead_ms,
            },

            # Performance verification vs attainment separation (PS 26053 Items 6 & 9)
            "performance_measurement": {
                "status": "VERIFIED",
                "method": "Stage-wise monotonic microsecond timers",
            },
            "performance_target": {
                "status": "NOT_MET_ON_CURRENT_CPU",
                "target_latency_ms": 30.0,
                "target_fps": 33.0,
                "measured_cpu_latency_ms": round(self.rolling_latency_ms, 2),
                "measured_cpu_fps": round(self.rolling_fps, 1),
                "cuda_status": model_status_dict.get("cuda_execution_provider", "NOT_AVAILABLE"),
                "note": "The current CPU implementation does not meet the PS 26053 latency/throughput target. GPU/CUDA/TensorRT acceleration should be benchmarked as the next optimisation path.",
            },

            # Memory metrics (binary MiB) & Scope (PS 26053 Items 1 & 5)
            "mapping_memory_mib": round(mapping_memory_mib, 2),
            "ram_mb": round(mapping_memory_mib, 2),
            "mapping_memory_target_mib": 15.0,
            "mapping_memory_target_status": "PASS" if mapping_memory_mib < 15.0 else "FAIL",
            "process_rss_mib": round(process_rss_mib, 2),
            "process_rss_scope": "informational only; not mapping memory",
            "model_file_size_mib": 7.36 if self.ml_adapter.salsa_session is not None else None,
            "model_runtime_memory_mib": "NOT_ISOLATED" if self.ml_adapter.salsa_session is not None else None,
            "model_memory_mib": None,  # Runtime model memory cannot be safely isolated from process RSS
            "gpu_vram_mib": self._last_vram_mb,

            # Theoretical reduction (Full-domain baseline: 120m x 100m @ 0.05m = 219.73 MiB)
            "uniform_5cm_full_domain_theoretical_memory_mib": round(uni_theo_mib, 2),
            "uniform_5cm_theoretical_mib": round(uni_theo_mib, 2),
            "adaptive_theoretical_mib": round(adap_theo_mib, 2),
            "theoretical_reduction_percent": round(theo_red_pct, 1),

            # Empirical benchmark (PS 26053 Items 1 & 2: Occupied-cell implementation memory)
            "uniform_5cm_empirical_occupied_mapping_memory_mib": uniform_measured_mib,
            "uniform_5cm_measured_mapping_memory_mib": uniform_measured_mib,
            "adaptive_empirical_mapping_memory_mib": adaptive_measured_mib,
            "adaptive_measured_mapping_memory_mib": adaptive_measured_mib,
            "empirical_memory_reduction_percent": emp_reduction_pct,
            "measured_reduction_percent": emp_reduction_pct,

            # Ground truth integrity (Section 1.2 & 13)
            "tracking_accuracy": None,
            "ground_truth_status": "GROUND TRUTH NOT AVAILABLE",

            # Legacy compatibility fields
            "system_status": health_info["system_status"],
            "sensor_health": health_info["status"],
            "degraded_quadrants": health_info["degraded_quadrants"],
            "point_count": len(clean_points),
            "ground_points": len(ground_pts),
            "obstacle_points": len(obstacle_pts),
            "ground_inlier_ratio": ground_inlier_ratio,
            "cell_count": len(leaves),
            "active_cells": len(leaves),
            "safe_cells": safe_count,
            "caution_cells": caution_count,
            "hazard_cells": hazard_count,
            "coarse_cells": tree_stats["coarse_cells"],
            "fine_cells": tree_stats["fine_cells"],
            "refinement_ratio": tree_stats["refinement_ratio"],
            "active_tracks": len(active_tracks),
            "dynamic_tracks": len(dynamic_tracks),
            "parked_cars": len(parked_car_clusters),
            "thin_hazards": len(thin_hazards),
            "negative_obstacles": len(dropoffs),
            "breakdown_ms": {
                "ingest": round(t_ingest, 2),
                "dust": round(t_dust, 2),
                "seg": round(t_seg, 2),
                "thin": round(t_thin, 2),
                "ml": round(t_ml, 2),
                "quadtree": round(t_quad, 2),
                "track": round(t_track, 2),
                "trench": round(t_trench, 2),
                "blend": round(t_blend, 2),
                "rollout": round(t_roll, 2),
                "costmap": round(t_cost, 2),
                "ros": round(t_ros, 2),
                "db": round(t_db, 2),
            },
            "mode": runtime_mode,
            "model_status": model_status_dict,
        }

        # 15. Serialize & Broadcast over WebSocket
        t_ser_start = time.perf_counter()
        payload = self.payload_builder.build_payload(
            frame_id=self.frame_count,
            timestamp=timestamp,
            system_stats=system_stats,
            leaves=leaves,
            tracks=active_tracks,
            hazard_cones=hazard_cones,
            parked_car_clusters=parked_car_clusters,
            raw_points=clean_points,
            mode=runtime_mode,
            model_status=model_status_dict,
        )
        t_ser = (time.perf_counter() - t_ser_start) * 1000.0
        system_stats["serialization_ms"] = round(t_ser, 2)
        system_stats["breakdown_ms"]["serialization"] = round(t_ser, 2)

        if not self.is_running:
            self.server.broadcast_nowait(payload)

        # 16. Database Telemetry
        self.telemetry_db.log_health(
            fps=self.rolling_fps,
            latency=self.rolling_latency_ms,
            cells=len(leaves),
            ram_mb=quadtree_ram_mb,
            vram_mb=self._last_vram_mb,
        )
        self.telemetry_db.log_tracks(active_tracks)

        # Periodic logging
        if self.frame_count % 100 == 0:
            health_msg = (
                f"[HEALTH CHECK] Frame: {self.frame_count:05d} | "
                f"FPS: {self.rolling_fps:5.1f} | "
                f"Latency: {self.rolling_latency_ms:5.2f} ms | "
                f"Active Leaves: {len(leaves)} | "
                f"Status: {health_info['system_status']}"
            )
            sys.stdout.write(health_msg + "\n")
            sys.stdout.flush()

        if self.frame_count % 25 == 0 or self.profile_mode:
            status_line = (
                f"Frame {self.frame_count:04d} | "
                f"FPS: {self.rolling_fps:4.1f} | "
                f"Latency: {self.rolling_latency_ms:4.1f}ms | "
                f"Pts: {len(clean_points)} (Obs: {len(obstacle_pts)}) | "
                f"Cells: {len(leaves)} (Fine: {tree_stats['fine_cells']}) | "
                f"Hazards: Thin={len(thin_hazards)}, Drop={len(dropoffs)}"
            )
            if self.profile_mode:
                b = system_stats.get("breakdown_ms")
                if isinstance(b, dict):
                    status_line += (
                        f"\n   [Profile Breakdown] Ingest: {b.get('ingest', 0)}ms | Dust: {b.get('dust', 0)}ms | "
                        f"Seg: {b.get('seg', 0)}ms | Thin: {b.get('thin', 0)}ms | Quad: {b.get('quadtree', 0)}ms | "
                        f"Track: {b.get('track', 0)}ms | Trench: {b.get('trench', 0)}ms | Blend: {b.get('blend', 0)}ms | "
                        f"Cost: {b.get('costmap', 0)}ms | ROS: {b.get('ros', 0)}ms"
                    )
            logger.info(status_line)

        return {
            "frame_id": self.frame_count,
            "timestamp": timestamp,
            "leaves": leaves,
            "quadtree_leaves": leaves,
            "tracks": active_tracks,
            "dynamic_tracks": dynamic_tracks,
            "hazard_cones": hazard_cones,
            "thin_hazards": thin_hazards,
            "dropoffs": dropoffs,
            "stats": system_stats,
            "system_stats": system_stats,
            "payload": payload,
            "ros_grid": ros_occupancy_grid,
        }

    async def run(
        self,
        frame_queue: Optional[asyncio.Queue] = None,
        max_frames: Optional[int] = None,
        pacing: bool = True,
    ):
        """
        Executes the asynchronous real-time perception loop.
        Listens on WebSocket and processes frames from frame_queue or network inputs.
        """
        logger.info(f"Initializing tactical perception pipeline. Target: {self.target_fps} Hz")
        if not self._warmed_up:
            self.warmup()
        if self.db_adapter.enabled:
            logger.info(f"Database Adapter enabled. Redis Connected: {self.db_adapter.is_connected}")
        if self.ros_output:
            logger.info(f"ROS 2 OccupancyGrid generation enabled ({NAV2.WIDTH}x{NAV2.HEIGHT}, res: {NAV2.RESOLUTION}m)")

        await self.server.start()
        self.is_running = True
        logger.info("Perception pipeline live server listening for incoming frames.")

        sweep_files: List[Path] = []

        # Load dynamic sequence sweeps from dataset_dir
        if self.dataset_dir and self.dataset_dir.exists() and self.dataset_dir.is_dir():
            sweep_files = sorted(list(self.dataset_dir.glob("*.bin")))
            if sweep_files:
                logger.info(f"Loaded dynamic sequence dataset from {self.dataset_dir}: {len(sweep_files)} sweeps")

        # Fallback to single static sweep if sequence directory is empty
        if not sweep_files:
            for candidate in [
                Path("data/test_sweep/000000.bin"),
                Path(__file__).resolve().parent.parent / "data" / "test_sweep" / "000000.bin",
                Path(__file__).resolve().parent.parent / "data" / "kitti_clean" / "training" / "velodyne" / "000000.bin",
            ]:
                if candidate.exists():
                    sweep_files = [candidate]
                    logger.info(f"Using single-frame LiDAR sweep fallback: {candidate}")
                    break

        sweep_idx = 0
        start_loop_time = time.perf_counter()

        try:
            while self.is_running:
                if self.is_paused or getattr(self.server, "is_paused", False):
                    if getattr(self, "_step_once", False):
                        self._step_once = False
                    else:
                        # Sleep lightly to yield CPU and prevent proxy timeout without pushing frames
                        await asyncio.sleep(0.05)
                        continue

                t0 = time.perf_counter()

                if frame_queue is not None:
                    try:
                        frame_data = await asyncio.wait_for(frame_queue.get(), timeout=0.1)
                        if frame_data is None:  # Sentinel to stop
                            break
                        if isinstance(frame_data, tuple):
                            pts, ts = frame_data[0], frame_data[1]
                            meta = frame_data[2] if len(frame_data) > 2 else None
                        elif isinstance(frame_data, dict):
                            pts = frame_data["points"]
                            ts = frame_data.get("timestamp", time.time())
                            meta = frame_data.get("meta", None)
                        else:
                            pts, ts, meta = frame_data, time.time(), None

                        # Offload heavy perception compute to thread pool
                        frame_result = await asyncio.to_thread(self.process_frame, pts, timestamp=ts, meta=meta)
                        if frame_result and "payload" in frame_result and self.server.connected_clients:
                            await self.server.broadcast(frame_result["payload"])
                        await asyncio.sleep(0.01)  # Explicitly yield control to socket event loop
                    except asyncio.TimeoutError:
                        continue
                    except Exception as frame_err:
                        logger.error(f"[Perception] Error in queue frame cycle: {frame_err}", exc_info=True)
                else:
                    pts = None
                    if getattr(self, "dataset_loader", None) and self.dataset_loader.files:
                        try:
                            pts = self.dataset_loader.get_next_sweep()
                        except Exception as e:
                            logger.warning(f"Error getting next sweep from dataset loader: {e}")
                            pts = None

                    if pts is None and sweep_files:
                        current_file = sweep_files[sweep_idx % len(sweep_files)]
                        sweep_idx += 1
                        try:
                            raw = np.fromfile(current_file, dtype=np.float32)
                            if raw.size % 4 == 0:
                                pts = raw.reshape(-1, 4)
                            elif raw.size % 5 == 0:
                                pts = raw.reshape(-1, 5)[:, :4]
                            else:
                                pts = raw.reshape(-1, 3)
                        except Exception as e:
                            logger.warning(f"Error reading sweep {current_file}: {e}")
                            pts = None

                    if pts is None:
                        pts = _generate_live_sensor_buffer(self.frame_count, dt=self.dt)

                    try:
                        # Offload heavy perception compute to thread pool
                        frame_result = await asyncio.to_thread(self.process_frame, pts, timestamp=time.time())
                        if frame_result and "payload" in frame_result and self.server.connected_clients:
                            await self.server.broadcast(frame_result["payload"])
                    except Exception as frame_err:
                        logger.error(f"[Perception] Error in perception cycle: {frame_err}", exc_info=True)
                    await asyncio.sleep(0.01)  # Explicitly yield control to socket event loop


                if max_frames is not None and self.frame_count >= max_frames:
                    logger.info(f"Target frame limit {max_frames} reached. Stopping.")
                    break

                # Adaptive non-blocking loop pacing to maintain target FPS without artificial blocking delay
                if pacing and not self.kaggle:
                    elapsed = time.perf_counter() - t0
                    speed_multiplier = getattr(self, "playback_speed", 1.0)
                    base_delay = 1.0 / max(0.5, self.target_fps)  # e.g., 1.0 / 4.0 = 0.25s
                    target_interval = base_delay / max(0.1, speed_multiplier)
                    actual_delay = max(0.05, target_interval - elapsed)
                    await asyncio.sleep(actual_delay)
                else:
                    await asyncio.sleep(0.001)

        except asyncio.CancelledError:
            logger.info("Perception loop cancelled.")
        finally:
            await self.server.stop()
            self.telemetry_db.close()
            total_time = time.perf_counter() - start_loop_time
            logger.info(
                f"Perception pipeline terminated. Processed {self.frame_count} frames "
                f"in {total_time:.2f}s."
            )

    def stop(self):
        """Signals loop termination."""
        self.is_running = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="SIH 2026 PS-53: Adaptive Variable Resolution 2.5D LiDAR Mapping Backend"
    )
    default_port = int(os.environ.get("PORT", CONFIG.WS_PORT))
    default_host = os.environ.get("HOST", CONFIG.WS_HOST)

    parser.add_argument(
        "--host",
        type=str,
        default=default_host,
        help="Host address to bind",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help="Port to bind server",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=float(os.environ.get("TARGET_FPS", 4.0)),
        help="Target loop frequency in Hz (default: 4.0)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to run before terminating (default: infinite)",
    )
    parser.add_argument(
        "--use-redis",
        action="store_true",
        help="Enable Redis database state synchronization for vehicle pose and tracks",
    )
    parser.add_argument(
        "--ros-output",
        action="store_true",
        help="Generate standard ROS 2 nav_msgs/msg/OccupancyGrid planar costmaps",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable high-resolution per-stage execution profiling in terminal",
    )
    parser.add_argument(
        "--dataset-dir",
        "--data-path",
        type=str,
        default=str(DEFAULT_DATASET_DIR),
        help=f"Directory containing sequential raw .bin point cloud sweeps (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="[Deprecated] Previously streamed sweeps remotely from Kaggle dataset API",
    )
    parser.add_argument(
        "--kaggle-dataset",
        type=str,
        default=None,
        help="[Deprecated] Kaggle dataset slug (<owner>/<dataset-name>)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize dataset loader directly from the bundled local files (zero network overhead, zero RAM spikes)
    data_loader = DatasetLoader(mode="dynamic")
    logger.info(f"[Main] Bundled sweep loader ready with {len(data_loader)} frames ({data_loader.data_dir})")

    # In main.py where PerceptionPipeline is instantiated:
    # Lower frequency (4.0 Hz) stops CPU throttling and prevents latency ballooning on Render's 0.2 vCPU
    TARGET_FPS = 4.0
    target_fps = args.fps if args.fps is not None else TARGET_FPS

    pipeline = PerceptionPipeline(
        host=args.host,
        port=args.port,
        target_fps=target_fps,
        use_redis=args.use_redis,
        ros_output=args.ros_output,
        profile_mode=args.profile,
        dataset_dir=data_loader.data_dir if data_loader.data_dir else args.dataset_dir,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def shutdown(sig_name="TERM"):
        logger.info(f"Received {sig_name} signal. Shutting down perception pipeline...")
        pipeline.stop()
        for task in asyncio.all_tasks(loop):
            task.cancel()

    signals_to_handle = [
        s for s in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None))
        if s is not None
    ]
    for sig in signals_to_handle:
        sig_name = getattr(sig, "name", str(sig))
        try:
            loop.add_signal_handler(sig, lambda name=sig_name: shutdown(name))
        except (NotImplementedError, AttributeError):
            try:
                signal.signal(sig, lambda s, f, name=sig_name: shutdown(name))
            except Exception:
                pass

    try:
        loop.run_until_complete(pipeline.run(max_frames=args.max_frames))
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Perception pipeline terminated cleanly.")
    finally:
        loop.close()


DRISHTIPerceptionPipeline = PerceptionPipeline
DrishtiEngine = PerceptionPipeline


if __name__ == "__main__":
    main()
