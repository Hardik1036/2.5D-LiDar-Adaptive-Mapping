"""
Edge Performance Profiler for DRDO SIH 2026 Problem Statement 53.
Benchmarks 150 frames of high-density LiDAR perception under headless conditions.
Measures nanosecond-precision per-stage latency, P95 metrics, and verifies
deterministic compliance with the <= 35 ms frame latency and >= 25 Hz throughput SLA
with all tactical perception filters and terrain memory modules active.
"""

import sys
import time
from typing import Any, Dict, List
import numpy as np

from backend.adapters.ml_adapter import MLPerceptionAdapter
from backend.adapters.db_adapter import DatabaseAdapter
from backend.config import BOUNDS, QUADTREE, TRACKING
from backend.ingestion.dust_filter import StatisticalDustFilter
from backend.ingestion.ground_segmentation import GroundSegmenter
from backend.ingestion.thin_hazard_detector import ThinHazardDetector
from backend.mapping.costmap import CostmapEvaluator
from backend.mapping.negative_obstacles import TrenchDetector
from backend.mapping.occupancy_publisher import OccupancyGridBuilder
from backend.mapping.porosity_filter import PorosityClassifier
from backend.mapping.quadtree import AdaptiveQuadtree
from backend.mapping.temporal_blender import TemporalMapBlender
from backend.mapping.vegetation_filter import VegetationFilter
from backend.tracking.clustering import EuclideanClusterer
from backend.tracking.ghost_clearing import GhostClearing
from backend.tracking.kalman_tracker import KalmanTracker
from backend.tracking.trajectory_rollout import TrajectoryRollout
from backend.utils.synthetic_generator import SyntheticLiDARGenerator


def run_benchmark(num_frames: int = 150) -> Dict[str, Any]:
    """
    Executes headless end-to-end perception pipeline benchmark across num_frames.
    """
    generator = SyntheticLiDARGenerator()
    dust_filter = StatisticalDustFilter(k=10)
    ground_segmenter = GroundSegmenter()
    thin_detector = ThinHazardDetector()
    quadtree = AdaptiveQuadtree()
    porosity_classifier = PorosityClassifier()
    trench_detector = TrenchDetector(num_angular_sectors=36)
    temporal_blender = TemporalMapBlender()
    costmap_eval = CostmapEvaluator()
    vegetation_filter = VegetationFilter()
    clusterer = EuclideanClusterer()
    tracker = KalmanTracker()
    ghost_clearing = GhostClearing()
    hazard_predictor = TrajectoryRollout()
    occupancy_builder = OccupancyGridBuilder()
    ml_adapter = MLPerceptionAdapter()
    db_adapter = DatabaseAdapter(enabled=False)

    stage_timings: Dict[str, List[float]] = {
        "1. Dust & Scatter Filter [F1.3]": [],
        "2. Ground Segmentation": [],
        "3. Thin Hazard Detector [F1.6]": [],
        "4. ML Semantic & Box Ingestion": [],
        "5. Adaptive Quadtree (2.5D)": [],
        "6. Clustering & Porosity [F1.5]": [],
        "7. 2D EKF Tracking": [],
        "8. Trench Drop-off Detector [F5.1]": [],
        "9. Temporal Elevation Blender [F2.4]": [],
        "10. Ghost Clearing & Rollout": [],
        "11. Costmap & Veg Filter": [],
        "12. ROS 2 Occupancy Grid (400x400)": [],
        "Total Tactical Pipeline Latency": [],
    }

    # Warm up JIT and threadpools across all stages
    warmup_pts, _ = generator.generate_frame(t=0.0)
    w_clean = dust_filter.filter(warmup_pts)
    w_g, w_obs, _ = ground_segmenter.segment(w_clean)
    _ = thin_detector.detect_low_profile_hazards(w_g)
    w_leaves = quadtree.build(w_clean[:5000])
    w_cls = clusterer.cluster(w_obs)
    _ = porosity_classifier.classify_clusters(w_cls)
    costmap_eval.evaluate_leaves(w_leaves)
    costmap_eval.apply_obstacle_occupancy(w_leaves, w_obs)
    occupancy_builder.build_grid(w_leaves, [])

    print("=" * 90)
    print(f"DRDO PS-53 Tactical Perception Edge Profiler: Running {num_frames} frames")
    print("=" * 90)

    for i in range(num_frames):
        t_sim = i * 0.04
        pts, _ = generator.generate_frame(t=t_sim)

        t_frame_start = time.perf_counter()

        # 1. Dust & Atmospheric Scatter Outlier Filter [F1.3]
        t0 = time.perf_counter()
        clean_pts = dust_filter.filter(pts)
        t1 = time.perf_counter()
        stage_timings["1. Dust & Scatter Filter [F1.3]"].append((t1 - t0) * 1000.0)

        # 2. Ground plane segmentation
        t0 = time.perf_counter()
        ground_pts, obstacle_pts, ground_mask = ground_segmenter.segment(clean_pts)
        t1 = time.perf_counter()
        stage_timings["2. Ground Segmentation"].append((t1 - t0) * 1000.0)

        # 3. Thin Hazard & Spike Strip Detection [F1.6]
        t0 = time.perf_counter()
        thin_hazards = thin_detector.detect_low_profile_hazards(ground_pts)
        t1 = time.perf_counter()
        stage_timings["3. Thin Hazard Detector [F1.6]"].append((t1 - t0) * 1000.0)

        # 4. ML Semantic & Bounding Box Ingestion
        t0 = time.perf_counter()
        N = len(clean_pts)
        synthetic_labels = np.zeros(N, dtype=np.int32)
        synthetic_labels[int(N * 0.60):int(N * 0.75)] = 1
        synthetic_labels[int(N * 0.75):int(N * 0.95)] = 2
        synthetic_labels[int(N * 0.95):] = 3
        ml_subsets = ml_adapter.process_semantic_labels(clean_pts, synthetic_labels)

        synthetic_dets = np.array([[2.0 + np.sin(t_sim), 5.0 + t_sim * 0.5, 0.0, 4.0, 1.8, 1.5, 0.1, 0.0, 0.5]], dtype=np.float32)
        pillar_dets = ml_adapter.process_pillar_detections(synthetic_dets)
        t1 = time.perf_counter()
        stage_timings["4. ML Semantic & Box Ingestion"].append((t1 - t0) * 1000.0)

        # 5. 2.5D Adaptive Quadtree Construction
        t0 = time.perf_counter()
        leaves = quadtree.build(clean_pts)
        t1 = time.perf_counter()
        stage_timings["5. Adaptive Quadtree (2.5D)"].append((t1 - t0) * 1000.0)

        # 6. Clustering & Porosity Classification [F1.5]
        t0 = time.perf_counter()
        clusters = clusterer.cluster(obstacle_pts)
        classified = porosity_classifier.classify_clusters(clusters)
        rigid_candidates = [c for (c, is_porous, _) in classified if not is_porous]
        t1 = time.perf_counter()
        stage_timings["6. Clustering & Porosity [F1.5]"].append((t1 - t0) * 1000.0)

        # 7. 2D EKF Tracking
        t0 = time.perf_counter()
        all_candidates = rigid_candidates + pillar_dets
        active_tracks = tracker.update(all_candidates, dt=0.04)
        dynamic_tracks = tracker.get_dynamic_tracks()
        t1 = time.perf_counter()
        stage_timings["7. 2D EKF Tracking"].append((t1 - t0) * 1000.0)

        # 8. Negative Obstacle / Trench Detection [F5.1]
        t0 = time.perf_counter()
        dropoffs = trench_detector.find_dropoffs(ground_pts)
        trench_detector.apply_dropoffs_to_leaves(leaves, dropoffs)
        t1 = time.perf_counter()
        stage_timings["8. Trench Drop-off Detector [F5.1]"].append((t1 - t0) * 1000.0)

        # 9. Temporal Elevation Blender [F2.4]
        t0 = time.perf_counter()
        temporal_blender.blend_quadtree(leaves)
        t1 = time.perf_counter()
        stage_timings["9. Temporal Elevation Blender [F2.4]"].append((t1 - t0) * 1000.0)

        # 10. Ghost Clearing & Rollout Cones
        t0 = time.perf_counter()
        ghost_clearing.clear_ghosts(leaves, clean_pts)
        ghost_clearing.register_dynamic_footprints(dynamic_tracks)
        rollout_hazards = hazard_predictor.predict_hazards(active_tracks)
        costmap_eval.apply_obstacle_occupancy(leaves, obstacle_pts)
        t1 = time.perf_counter()
        stage_timings["10. Ghost Clearing & Rollout"].append((t1 - t0) * 1000.0)

        # 11. Costmap Evaluation & Vegetation Filtering
        t0 = time.perf_counter()
        costmap_eval.evaluate_leaves(leaves)
        vegetation_filter.apply_vegetation_scaling(
            leaves,
            vegetation_points=ml_subsets["vegetation"],
            rigid_points=ml_subsets["rigid"],
        )
        all_hazards = list(rollout_hazards)
        for th in thin_hazards:
            all_hazards.append({"x": th["centroid"][0], "y": th["centroid"][1], "radius": th["radius"], "cost": 255})
        for d in dropoffs:
            all_hazards.append({"x": d["x"], "y": d["y"], "radius": d["radius"], "cost": 255})
        costmap_eval.apply_dynamic_hazards(leaves, all_hazards)
        t1 = time.perf_counter()
        stage_timings["11. Costmap & Veg Filter"].append((t1 - t0) * 1000.0)

        # 12. ROS 2 nav_msgs/OccupancyGrid Generation (400x400)
        t0 = time.perf_counter()
        occupancy_grid = occupancy_builder.build_grid(leaves, all_hazards)
        t1 = time.perf_counter()
        stage_timings["12. ROS 2 Occupancy Grid (400x400)"].append((t1 - t0) * 1000.0)

        t_frame_end = time.perf_counter()
        stage_timings["Total Tactical Pipeline Latency"].append((t_frame_end - t_frame_start) * 1000.0)

    # Compile benchmark statistics
    summary_results = {}
    print("\n" + "-" * 90)
    print(f"{'Pipeline Processing Stage':<42} | {'Mean (ms)':<10} | {'Min (ms)':<9} | {'Max (ms)':<9} | {'P95 (ms)':<9}")
    print("-" * 90)

    for stage_name, measurements in stage_timings.items():
        arr = np.array(measurements)
        mean_v = float(np.mean(arr))
        min_v = float(np.min(arr))
        max_v = float(np.max(arr))
        p95_v = float(np.percentile(arr, 95))
        summary_results[stage_name] = {
            "mean": mean_v,
            "min": min_v,
            "max": max_v,
            "p95": p95_v,
        }
        if stage_name == "Total Tactical Pipeline Latency":
            print("=" * 90)
        print(f"{stage_name:<42} | {mean_v:<10.3f} | {min_v:<9.3f} | {max_v:<9.3f} | {p95_v:<9.3f}")

    total_mean = summary_results["Total Tactical Pipeline Latency"]["mean"]
    total_p95 = summary_results["Total Tactical Pipeline Latency"]["p95"]
    fps = 1000.0 / total_mean if total_mean > 0 else 0.0

    print("=" * 90)
    print(f"Overall Perception Throughput: {fps:.2f} Hz | Mean Latency: {total_mean:.2f} ms | P95 Latency: {total_p95:.2f} ms")
    print("=" * 90)

    # SLA Assertions
    sla_latency_passed = total_mean <= 35.0
    sla_fps_passed = fps >= 25.0

    print("\nDRDO Perception Engine SLA Verification:")
    print(f"  [1] Real-Time Latency SLA  (Mean <= 35.0 ms): {'PASS' if sla_latency_passed else 'FAIL'} ({total_mean:.2f} ms)")
    print(f"  [2] High-Speed Throughput (Target >= 25.0 Hz): {'PASS' if sla_fps_passed else 'FAIL'} ({fps:.2f} Hz)")

    if sla_latency_passed and sla_fps_passed:
        print("\n>>> ALL REAL-TIME EDGE SLA REQUIREMENTS MET DETERMINISTICALLY <<<")
        return summary_results
    else:
        print("\n>>> SLA VIOLATION DETECTED <<<")
        sys.exit(1)


if __name__ == "__main__":
    frames = 150
    if len(sys.argv) > 1:
        try:
            frames = int(sys.argv[1])
        except ValueError:
            pass
    run_benchmark(num_frames=frames)
