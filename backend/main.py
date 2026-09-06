"""
Pipeline Orchestrator for SIH 2026 Problem Statement 53:
DRDO: Adaptive Variable Resolution 2.5D LiDAR Mapping for Dynamic Perception.

Real-time Tactical Loop:
  1. Ingest Point Cloud (Binary / PCD / Synthetic)
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

import argparse
import asyncio
import logging
import os
from pathlib import Path
import signal
import sys
import time
from typing import Optional

import numpy as np

from backend.adapters.db_adapter import DatabaseAdapter
from backend.adapters.ml_adapter import MLPerceptionAdapter
from backend.config import BOUNDS, CONFIG, COSTMAP, DATABASE, NAV2, QUADTREE, SERVER, TRACKING
from backend.ingestion.dataset_loader import DatasetLoader
from backend.ingestion.dust_filter import StatisticalDustFilter
from backend.ingestion.ground_segmentation import GroundSegmenter
from backend.ingestion.thin_hazard_detector import ThinHazardDetector
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


class PerceptionPipeline:
    """
    Coordinates the full real-time tactical LiDAR perception stack.
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        host: str = SERVER.HOST,
        port: int = SERVER.PORT,
        target_fps: float = SERVER.TARGET_FPS,
        use_redis: bool = False,
        ros_output: bool = False,
        profile_mode: bool = False,
    ):
        self.target_fps = target_fps
        self.dt = 1.0 / target_fps
        self.is_running = False
        self.ros_output = ros_output
        self.profile_mode = profile_mode

        # Ingestion data-path validation & automatic fallback to SyntheticScanGenerator
        resolved_data_path = None
        if dataset_path:
            p = Path(dataset_path)
            if p.exists() and (p.is_file() or (p.is_dir() and any(p.glob("*.*")))):
                resolved_data_path = str(p)
                logger.info(f"Ingestion source verified: '{resolved_data_path}'")
            else:
                logger.warning(
                    f"Provided data path '{dataset_path}' is invalid, does not exist, or contains no files. "
                    f"Automatically falling back to SyntheticScanGenerator for 24/7 continuous stream."
                )
        else:
            logger.info("No --data-path specified. Streaming 24/7 from SyntheticScanGenerator.")

        # Ingestion & Tactical Ingestion Filters
        self.loader = DatasetLoader(dataset_path=resolved_data_path, loop=True, use_synthetic_fallback=True)
        self.dust_filter = StatisticalDustFilter()
        self.ground_seg = GroundSegmenter()
        self.thin_hazard_detector = ThinHazardDetector()

        # ML Integration & Database Adapters
        self.ml_adapter = MLPerceptionAdapter()
        self.db_adapter = DatabaseAdapter(enabled=use_redis)

        # 2.5D Adaptive Quadtree Mapping, Terrain Memory, & Tactical Evaluators
        self.quadtree = AdaptiveQuadtree()
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

        # Metrics
        self.frame_count = 0
        self.rolling_fps = 0.0
        self.rolling_latency_ms = 0.0

    async def run(self, max_frames: Optional[int] = None, pacing: bool = True):
        """Executes the asynchronous real-time perception loop."""
        logger.info(f"Initializing tactical perception pipeline. Target: {self.target_fps} Hz")
        if self.db_adapter.enabled:
            logger.info(f"Database Adapter enabled. Redis Connected: {self.db_adapter.is_connected}")
        if self.ros_output:
            logger.info(f"ROS 2 OccupancyGrid generation enabled ({NAV2.WIDTH}x{NAV2.HEIGHT}, res: {NAV2.RESOLUTION}m)")

        await self.server.start()
        self.is_running = True

        alpha = 0.1  # Exponential moving average factor for stats
        start_loop_time = time.perf_counter()

        try:
            while self.is_running:
                t0 = time.perf_counter()
                self.frame_count += 1

                # 1. Ingestion
                t_ingest_start = time.perf_counter()
                raw_points, meta = self.loader.load_frame()
                t_ingest = (time.perf_counter() - t_ingest_start) * 1000.0

                # 2. Statistical Dust Outlier Filtering [F1.3]
                t_dust_start = time.perf_counter()
                clean_points = self.dust_filter.filter(raw_points)
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

                # 5. ML Perception Adapter (Semantic label partitioning & Pillar detections)
                t_ml_start = time.perf_counter()
                semantic_labels = meta.get("semantic_labels") if isinstance(meta, dict) else None
                ml_subsets = self.ml_adapter.process_semantic_labels(clean_points, semantic_labels)
                pillar_dets = self.ml_adapter.process_pillar_detections(
                    meta.get("dynamic_detections") if isinstance(meta, dict) else None
                )
                t_ml = (time.perf_counter() - t_ml_start) * 1000.0

                # 6. Adaptive 2.5D Quadtree construction
                t_quad_start = time.perf_counter()
                leaves = self.quadtree.build(clean_points)
                # Mark detected spike/hazard coordinates directly into 2.5D Quadtree leaf costs (cost = 255)
                if thin_hazards:
                    self.thin_hazard_detector.apply_hazards_to_leaves(leaves, thin_hazards)
                t_quad = (time.perf_counter() - t_quad_start) * 1000.0

                # 7. Obstacle clustering & Porosity Classification [F1.5]
                t_track_start = time.perf_counter()
                clusters = self.clusterer.cluster(obstacle_pts)
                classified_clusters = self.porosity_classifier.classify_clusters(clusters)
                
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

                # 10. Ghost smear clearing & trajectory rollout
                t_roll_start = time.perf_counter()
                self.ghost_clearing.clear_ghosts(leaves, clean_points)
                self.ghost_clearing.register_dynamic_footprints(dynamic_tracks)
                hazard_cones = self.rollout.rollout_all(active_tracks)
                t_roll = (time.perf_counter() - t_roll_start) * 1000.0

                # 11. Traversability costmap evaluation & vegetation scaling
                t_cost_start = time.perf_counter()
                self.costmap.evaluate_leaves(leaves)
                self.costmap.apply_obstacle_occupancy(leaves, obstacle_pts)
                
                # Apply vegetation filtering
                if len(ml_subsets["vegetation"]) > 0:
                    self.vegetation_filter.apply_vegetation_scaling(
                        leaves,
                        vegetation_points=ml_subsets["vegetation"],
                        rigid_points=ml_subsets["rigid"],
                    )

                # Collect all hazards (limit to closest 5 dynamic threat cones for <= 35ms SLA)
                rollout_hazards = self.rollout.predict_hazards(active_tracks, max_hazards=5)
                all_hazards = list(rollout_hazards)
                for th in thin_hazards:
                    all_hazards.append({"x": th["centroid"][0], "y": th["centroid"][1], "radius": th["radius"], "cost": 255})
                for d in dropoffs:
                    all_hazards.append({"x": d["x"], "y": d["y"], "radius": d["radius"], "cost": 255})

                self.costmap.apply_dynamic_hazards(leaves, all_hazards)

                # 12. Degraded-Mode Safe Fallback [F5.3]
                if health_info["is_degraded"]:
                    self.sensor_health.apply_degraded_costmap_inflation(leaves, caution_cost=180)
                    self.sensor_health.preserve_terrain_memory(leaves, self.temporal_blender)

                t_cost = (time.perf_counter() - t_cost_start) * 1000.0

                # 13. ROS 2 nav_msgs/OccupancyGrid Generation (optional)
                t_ros_start = time.perf_counter()
                ros_occupancy_grid = None
                if self.ros_output and self.occupancy_builder is not None:
                    grid_matrix = self.occupancy_builder.build_grid(leaves, all_hazards)
                    ros_occupancy_grid = self.occupancy_builder.to_ros_message_dict(grid_matrix, timestamp=time.time())
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

                if self.rolling_fps == 0.0:
                    self.rolling_fps = instant_fps
                    self.rolling_latency_ms = total_latency_ms
                else:
                    self.rolling_fps = (1 - alpha) * self.rolling_fps + alpha * instant_fps
                    self.rolling_latency_ms = (1 - alpha) * self.rolling_latency_ms + alpha * total_latency_ms

                tree_stats = self.quadtree.get_statistics()
                system_stats = {
                    "fps": round(self.rolling_fps, 1),
                    "latency_ms": round(self.rolling_latency_ms, 2),
                    "system_status": health_info["system_status"],
                    "sensor_health": health_info["status"],
                    "degraded_quadrants": health_info["degraded_quadrants"],
                    "point_count": len(clean_points),
                    "ground_points": len(ground_pts),
                    "obstacle_points": len(obstacle_pts),
                    "cell_count": len(leaves),
                    "coarse_cells": tree_stats["coarse_cells"],
                    "fine_cells": tree_stats["fine_cells"],
                    "refinement_ratio": tree_stats["refinement_ratio"],
                    "active_tracks": len(active_tracks),
                    "dynamic_tracks": len(dynamic_tracks),
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

                # 14. Serialize & Broadcast over WebSocket
                payload = self.payload_builder.build_payload(
                    frame_id=self.frame_count,
                    timestamp=time.time(),
                    system_stats=system_stats,
                    leaves=leaves,
                    tracks=active_tracks,
                    hazard_cones=hazard_cones,
                )
                # Non-blocking concurrent broadcast (never stalls perception loop)
                self.server.broadcast_nowait(payload)

                # Cloud health telemetry logging to stdout every 100 frames
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

                # Periodic terminal telemetry logging (every 25 frames ~ 1 sec)
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

                if max_frames is not None and self.frame_count >= max_frames:
                    logger.info(f"Target frame limit {max_frames} reached. Stopping.")
                    break

                # Frame pacing to maintain target loop rate
                if pacing:
                    elapsed = time.perf_counter() - t0
                    sleep_time = max(0.0, self.dt - elapsed)
                    await asyncio.sleep(sleep_time)
                else:
                    await asyncio.sleep(0.0005)

        except asyncio.CancelledError:
            logger.info("Perception loop cancelled.")
        finally:
            await self.server.stop()
            total_time = time.perf_counter() - start_loop_time
            logger.info(
                f"Perception pipeline terminated. Processed {self.frame_count} frames "
                f"in {total_time:.2f}s (Avg FPS: {self.frame_count / max(total_time, 0.001):.1f})."
            )

    def stop(self):
        """Signals loop termination."""
        self.is_running = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="SIH 2026 PS-53: Adaptive Variable Resolution 2.5D LiDAR Mapping Backend"
    )
    parser.add_argument(
        "--data-path",
        "--dataset",
        dest="dataset",
        type=str,
        default=os.environ.get("DATA_PATH", None),
        help="Path to .bin file, .pcd file, or dataset directory. Falls back to SyntheticScanGenerator if missing or invalid.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=CONFIG.WS_HOST,
        help=f"WebSocket server host (default: {CONFIG.WS_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=CONFIG.WS_PORT,
        help=f"WebSocket server port (default: {CONFIG.WS_PORT})",
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
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = PerceptionPipeline(
        dataset_path=args.dataset,
        host=args.host,
        port=args.port,
        target_fps=args.fps,
        use_redis=args.use_redis,
        ros_output=args.ros_output,
        profile_mode=args.profile,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def shutdown(sig_name="TERM"):
        logger.info(f"Received {sig_name} signal. Shutting down perception pipeline...")
        pipeline.stop()
        for task in asyncio.all_tasks(loop):
            task.cancel()

    # Clean signal handling across POSIX containers (Docker / Kubernetes / Railway / Render) and Windows
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


if __name__ == "__main__":
    main()
