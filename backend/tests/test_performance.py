"""
Performance SLA regression test for DRDO SIH 2026 Problem Statement 53.
Asserts <= 35 ms latency and >= 25 Hz throughput over 25 consecutive frames
with all advanced tactical perception modules and terrain memory enabled.
"""

import time
import numpy as np
import pytest

from backend.adapters.ml_adapter import MLPerceptionAdapter
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


def test_realtime_performance_sla():
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

    latencies = []
    num_frames = 30

    # Complete warmup across all modules (JIT, threadpools, Cython allocations)
    p0, _ = generator.generate_frame(0.0)
    w_clean = dust_filter.filter(p0)
    w_g, w_obs, _ = ground_segmenter.segment(w_clean)
    _ = thin_detector.detect_low_profile_hazards(w_g)
    w_leaves = quadtree.build(w_clean[:5000])
    w_cls = clusterer.cluster(w_obs)
    _ = porosity_classifier.classify_clusters(w_cls)
    costmap_eval.evaluate_leaves(w_leaves)
    costmap_eval.apply_obstacle_occupancy(w_leaves, w_obs)
    occupancy_builder.build_grid(w_leaves, [])

    for i in range(num_frames):
        t_sim = i * 0.04
        pts, _ = generator.generate_frame(t=t_sim)

        t_start = time.perf_counter()

        # 1. Dust & atmospheric scatter filter [F1.3]
        clean_pts = dust_filter.filter(pts)

        # 2. Ground plane segmentation
        ground_pts, obstacle_pts, ground_mask = ground_segmenter.segment(clean_pts)

        # 3. Thin hazard & spike strip detection [F1.6]
        thin_hazards = thin_detector.detect_low_profile_hazards(ground_pts)

        # 4. Adaptive Quadtree build
        leaves = quadtree.build(clean_pts)
        costmap_eval.evaluate_leaves(leaves)

        # 5. Semantic adapter & vegetation scaling
        N = len(clean_pts)
        synthetic_labels = np.zeros(N, dtype=np.int32)
        synthetic_labels[int(N * 0.7):] = 1
        ml_subsets = ml_adapter.process_semantic_labels(clean_pts, synthetic_labels)
        vegetation_filter.apply_vegetation_scaling(leaves, ml_subsets["vegetation"])

        # 6. Obstacle clustering & Porosity Classification [F1.5]
        clusters = clusterer.cluster(obstacle_pts)
        classified = porosity_classifier.classify_clusters(clusters)
        rigid_candidates = [c for (c, is_porous, _) in classified if not is_porous]

        # 7. Tracking
        active_tracks = tracker.update(rigid_candidates, dt=0.04)
        dynamic_tracks = tracker.get_dynamic_tracks()

        # 8. Negative obstacle / Trench detection [F5.1]
        dropoffs = trench_detector.find_dropoffs(ground_pts)
        trench_detector.apply_dropoffs_to_leaves(leaves, dropoffs)

        # 9. Temporal elevation blending [F2.4]
        temporal_blender.blend_quadtree(leaves)

        # 10. Ghost clearing & obstacle occupancy
        ghost_clearing.clear_ghosts(leaves, clean_pts)
        ghost_clearing.register_dynamic_footprints(dynamic_tracks)
        costmap_eval.apply_obstacle_occupancy(leaves, obstacle_pts)

        # 11. Dynamic rollout & ROS 2 Occupancy Grid
        hazards = hazard_predictor.predict_hazards(active_tracks)
        for th in thin_hazards:
            hazards.append({"x": th["centroid"][0], "y": th["centroid"][1], "radius": th["radius"], "cost": 255})
        for d in dropoffs:
            hazards.append({"x": d["x"], "y": d["y"], "radius": d["radius"], "cost": 255})

        costmap_eval.apply_dynamic_hazards(leaves, hazards)
        _ = occupancy_builder.build_grid(leaves, hazards)

        t_end = time.perf_counter()
        latencies.append((t_end - t_start) * 1000.0)

    stable_latencies = latencies[2:]
    mean_latency = float(np.mean(stable_latencies))
    fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0

    print(f"\n[Test Performance Full Tactical Perception] Mean Latency: {mean_latency:.2f} ms | Effective FPS: {fps:.2f} Hz")

    assert mean_latency <= 35.0, f"Mean latency {mean_latency:.2f} ms violated SLA (<= 35 ms)"
    assert fps >= 25.0, f"Throughput {fps:.2f} Hz violated SLA (>= 25 Hz)"
