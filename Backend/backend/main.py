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
    import psutil
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
from backend.ingestion.dataset_loader import DatasetLoader
from backend.mapping.costmap import CostmapEvaluator
from backend.mapping.degraded_mode import SensorHealthMonitor
from backend.mapping.negative_obstacles import TrenchDetector
from backend.mapping.occupancy_publisher import OccupancyGridBuilder
from backend.mapping.porosity_filter import PorosityClassifier
from backend.mapping.quadtree import AdaptiveQuadtree
from backend.mapping.temporal_blender import TemporalMapBlender
from backend.mapping.vegetation_filter import VegetationFilter
from backend.server.payload_builder import PayloadBuilder
from backend.server.websocket_server import TelemetryWebSocketServer
from backend.telemetry.telemetry_db import AsyncTelemetryDB
from backend.tracking.clustering import EuclideanClusterer
from backend.tracking.ghost_clearing import GhostClearing
from backend.tracking.kalman_tracker import KalmanTracker
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
        kaggle_dataset: Optional[str] = "hardiknopany1036/drishti-2-5d-dataset",
        kaggle_streamer: Optional[Any] = None,
        **kwargs,
    ):
        self.config = CONFIG
        self.target_fps = target_fps
        self.dt = 1.0 / target_fps
        self.is_running = False
        self.ros_output = ros_output
        self.profile_mode = profile_mode
        self.kaggle = kaggle or (kaggle_streamer is not None)
        self.kaggle_dataset = kaggle_dataset or "hardiknopany1036/drishti-2-5d-dataset"
        self.kaggle_streamer = kaggle_streamer
        ds_path = Path(dataset_dir) if dataset_dir is not None else DEFAULT_DATASET_DIR
        # Ensure path points strictly to the full sweeps in training/velodyne, never parent or gt_database
        if not self.kaggle and ds_path.is_dir():
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
        self.payload_builder = PayloadBuilder()
        self.telemetry_db = AsyncTelemetryDB()

        # Dataset loader & dynamic sequence routing
        self.dataset_loader = DatasetLoader(self.dataset_dir)
        self.loader = self.dataset_loader
        self.current_dataset_mode = "static"

        # Dynamic Playback Controls (HUD WebSocket listener)
        self.is_paused = False
        self.playback_speed = 1.0
        self.server.set_command_callback(self.handle_client_command)

        # Metrics
        self.frame_count = 0
        self.rolling_fps = 0.0
        self.rolling_latency_ms = 0.0
        self._last_ram_mb: Optional[float] = None
        self._last_vram_mb: Optional[float] = None
        self._warmed_up = False

    def handle_client_command(self, cmd_data: dict) -> None:
        """
        Processes dynamic runtime control commands received from connected frontend HUD clients.
        Supported actions:
          - 'pause': Halts sweep progression without closing WebSocket or dropping client state
          - 'resume' / 'play': Resumes sweep loop execution
          - 'toggle_pause': Inverts current pause state
          - 'set_speed': Scales frame pacing interval (e.g. 0.25x, 0.5x, 1.0x, 2.0x)
        """
        if not isinstance(cmd_data, dict):
            return
        action = cmd_data.get("action") or cmd_data.get("command") or cmd_data.get("type")
        if not action:
            return
        action = str(action).lower().strip()
        if action == "pause":
            self.is_paused = True
            logger.info("Pipeline playback PAUSED via client command.")
        elif action in ("resume", "play"):
            self.is_paused = False
            logger.info("Pipeline playback RESUMED via client command.")
        elif action == "toggle_pause":
            self.is_paused = not self.is_paused
            logger.info(f"Pipeline playback pause toggled: {self.is_paused}")
        elif action == "set_speed":
            try:
                speed = float(cmd_data.get("speed") or cmd_data.get("value", 1.0))
                self.playback_speed = max(0.1, min(10.0, speed))
                logger.info(f"Pipeline playback speed set to: {self.playback_speed:.2f}x")
            except (ValueError, TypeError):
                pass
        elif action == "set_dataset":
            data = cmd_data
            target_mode = str(data.get("mode", "static")).lower()
            search_roots = [
                Path("data/kaggle_cache"),
                Path("data"),
                Path(__file__).resolve().parent.parent / "data" / "kaggle_cache",
                Path(__file__).resolve().parent.parent / "data",
            ]

            if target_mode == "dynamic":
                # Match dynamic tracking sequences
                candidates = []
                for root in search_roots:
                    if root.exists():
                        candidates.extend(list(root.rglob("*dynamic*/**/velodyne")))
                        candidates.extend(list(root.rglob("*tracking*/**/velodyne")))
                        candidates.extend(list(root.rglob("*dynamic*")))

                target_dir = next(
                    (c for c in candidates if c.is_dir() and any(c.glob("*.bin"))),
                    None,
                )
                desc = "Continuous Dynamic Multi-Object Tracking"
            else:
                # Match clean/static baseline sequences
                candidates = []
                for root in search_roots:
                    if root.exists():
                        candidates.extend(list(root.rglob("*clean*/**/velodyne")))
                        candidates.extend(list(root.rglob("*clean*")))
                        candidates.append(root / "kitti_clean" / "training" / "velodyne")

                target_dir = next(
                    (c for c in candidates if c.is_dir() and any(c.glob("*.bin"))),
                    None,
                )
                desc = "Clean Static Urban Road Corridor"

            if target_dir and target_dir.exists():
                logger.info(f"[Perception] Switching dataset stream to: {desc} ({target_dir})")
                swapped = False
                for attr_name in ['loader', 'dataset_loader', 'streamer', 'kaggle_streamer']:
                    loader_instance = getattr(self, attr_name, None)
                    if loader_instance and hasattr(loader_instance, 'set_directory'):
                        swapped = loader_instance.set_directory(str(target_dir))
                        break

                if swapped:
                    self.current_dataset_mode = target_mode
                    self.dataset_dir = target_dir
                    if hasattr(self, 'server') and self.server:
                        try:
                            loop = asyncio.get_running_loop()
                            loop.create_task(
                                self.server.broadcast(json.dumps({
                                    "type": "dataset_swapped",
                                    "mode": target_mode,
                                    "description": desc
                                }))
                            )
                        except RuntimeError:
                            pass
            else:
                logger.warning(f"[Perception] Could not locate candidate directory for mode: {target_mode}")

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
    ) -> Dict[str, Any]:
        """
        Primary entry point for live point cloud ingestion.
        Processes point arrays passed directly as input arguments in memory.

        Args:
            points: [N, 3], [N, 4], or [N, 5] numpy float array of LiDAR coordinates.
            timestamp: Sensor acquisition timestamp in seconds (defaults to current time).
            meta: Optional metadata dictionary (e.g. semantic labels, dynamic detections).

        Returns:
            Dictionary containing processed frame results, leaves, tracks, and telemetry.
        """
        t0 = time.perf_counter()
        if timestamp is None:
            timestamp = time.time()
        if meta is None:
            meta = {}

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

        # 6. Adaptive 2.5D Quadtree construction
        t_quad_start = time.perf_counter()
        point_costs = np.zeros(len(clean_points), dtype=np.int32)
        if len(threat_points) > 0:
            ground_indices = np.nonzero(semantic_labels == 0)[0]
            point_costs[ground_indices[threat_mask]] = 255

        leaves = self.quadtree.build(clean_points, point_costs=point_costs)
        leaves = self.quadtree.interpolate_ground_rings()
        # leaves = self.quadtree.interpolate_obstacle_clusters()  # Bypassed: map only real physical obstacle returns
        leaves = self.quadtree.apply_clearance_coarsening()
        # Mark detected spike/hazard coordinates directly into 2.5D Quadtree leaf costs (cost = 255)
        if thin_hazards:
            self.thin_hazard_detector.apply_hazards_to_leaves(leaves, thin_hazards)
        t_quad = (time.perf_counter() - t_quad_start) * 1000.0

        # 7. Obstacle clustering & Porosity Classification [F1.5]
        t_track_start = time.perf_counter()
        if len(obstacle_pts) > 10000:
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

        system_stats = {
            "fps": round(self.rolling_fps, 1),
            "latency_ms": round(self.rolling_latency_ms, 2),
            "system_status": health_info["system_status"],
            "sensor_health": health_info["status"],
            "degraded_quadrants": health_info["degraded_quadrants"],
            "point_count": len(clean_points),
            "ground_points": len(ground_pts),
            "obstacle_points": len(obstacle_pts),
            "ground_inlier_ratio": ground_inlier_ratio,
            "ram_mb": quadtree_ram_mb,
            "cell_count": len(leaves),
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
            }
        }

        # Dynamic Perception Fidelity reflecting ground inliers & Kalman innovation
        system_stats["tracking_accuracy"] = self.payload_builder.compute_tracking_accuracy(
            active_tracks, system_stats, leaves
        )

        # 15. Serialize & Broadcast over WebSocket
        payload = self.payload_builder.build_payload(
            frame_id=self.frame_count,
            timestamp=timestamp,
            system_stats=system_stats,
            leaves=leaves,
            tracks=active_tracks,
            hazard_cones=hazard_cones,
            parked_car_clusters=parked_car_clusters,
            raw_points=raw_points,
        )
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
                b = system_stats["breakdown_ms"]
                status_line += (
                    f"\n   [Profile Breakdown] Ingest: {b['ingest']}ms | Dust: {b['dust']}ms | "
                    f"Seg: {b['seg']}ms | Thin: {b['thin']}ms | Quad: {b['quadtree']}ms | "
                    f"Track: {b['track']}ms | Trench: {b['trench']}ms | Blend: {b['blend']}ms | "
                    f"Cost: {b['costmap']}ms | ROS: {b['ros']}ms"
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

        kaggle_gen = None
        sweep_files: List[Path] = []

        if self.kaggle:
            logger.info(f"[KaggleStreamer] Kaggle remote streaming enabled: {self.kaggle_dataset}")
            if self.kaggle_streamer is None:
                from backend.ingestion.kaggle_streamer import KaggleDatasetStreamer
                self.kaggle_streamer = KaggleDatasetStreamer(
                    dataset_slug=self.kaggle_dataset,
                    loop=True,
                )
            kaggle_gen = self.kaggle_streamer.stream(fps=self.target_fps)
            logger.info(f"[KaggleStreamer] Initialized with {len(self.kaggle_streamer.sweep_files)} sweeps ready for streaming.")
        else:
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
                if self.is_paused:
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
                        if frame_result and "payload" in frame_result:
                            await self.server.broadcast(frame_result["payload"])
                        await asyncio.sleep(0.01)  # Explicitly yield control to socket event loop
                    except asyncio.TimeoutError:
                        continue
                elif self.kaggle and kaggle_gen is not None:
                    try:
                        pts, meta = await asyncio.to_thread(next, kaggle_gen)
                    except StopIteration:
                        logger.info("Kaggle stream reached end of sequence.")
                        break
                    ts = meta.get("timestamp", time.time()) if meta else time.time()
                    # Offload heavy perception compute to thread pool
                    frame_result = await asyncio.to_thread(self.process_frame, pts, timestamp=ts, meta=meta)
                    if frame_result and "payload" in frame_result:
                        await self.server.broadcast(frame_result["payload"])
                    await asyncio.sleep(0.01)  # Explicitly yield control to socket event loop
                else:
                    active_files = (
                        self.dataset_loader.files
                        if (getattr(self, "dataset_loader", None) and self.dataset_loader.files)
                        else sweep_files
                    )
                    if active_files:
                        current_file = active_files[sweep_idx % len(active_files)]
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
                            pts = _generate_live_sensor_buffer(self.frame_count, dt=self.dt)
                    else:
                        pts = _generate_live_sensor_buffer(self.frame_count, dt=self.dt)

                    # Offload heavy perception compute to thread pool
                    frame_result = await asyncio.to_thread(self.process_frame, pts, timestamp=time.time())
                    if frame_result and "payload" in frame_result:
                        await self.server.broadcast(frame_result["payload"])
                    await asyncio.sleep(0.01)  # Explicitly yield control to socket event loop


                if max_frames is not None and self.frame_count >= max_frames:
                    logger.info(f"Target frame limit {max_frames} reached. Stopping.")
                    break

                # Adaptive non-blocking loop pacing to maintain target FPS without artificial blocking delay
                if pacing and not self.kaggle:
                    elapsed = time.perf_counter() - t0
                    base_interval = 1.0 / max(1.0, self.target_fps)
                    target_interval = base_interval / max(0.1, self.playback_speed)
                    sleep_time = max(0.001, target_interval - elapsed)
                    await asyncio.sleep(sleep_time)
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
        default=CONFIG.TARGET_FPS,
        help=f"Target loop frequency in Hz (default: {CONFIG.TARGET_FPS})",
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
        help="Stream sweeps remotely from Kaggle dataset API",
    )
    parser.add_argument(
        "--kaggle-dataset",
        type=str,
        default="hardiknopany1036/drishti-2-5d-dataset",
        help="Kaggle dataset slug (<owner>/<dataset-name>)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = PerceptionPipeline(
        host=args.host,
        port=args.port,
        target_fps=args.fps,
        use_redis=args.use_redis,
        ros_output=args.ros_output,
        profile_mode=args.profile,
        dataset_dir=args.dataset_dir,
        kaggle=args.kaggle,
        kaggle_dataset=args.kaggle_dataset,
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


if __name__ == "__main__":
    main()
