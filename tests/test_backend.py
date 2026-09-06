"""
Unit and Integration Tests for SIH 2026 Problem Statement 53 Backend.
Tests:
  - Configuration bounds and thresholds
  - Synthetic LiDAR generator output and dimensions
  - Ground plane vs non-ground segmentation
  - Adaptive 2.5D Quadtree construction and selective subdivision
  - Compact Cell Statistics calculation
  - Traversability Costmap evaluation
  - Euclidean Clustering on non-ground points
  - 2D Constant Velocity Kalman Tracker & Hungarian association
  - Trajectory Rollout and expanding hazard cones
  - Full Perception Pipeline execution (50 frames) verifying >= 25 Hz throughput (< 35ms latency)
"""

import asyncio
import time
import numpy as np
import pytest

from backend.config import BOUNDS, COSTMAP, QUADTREE, TRACKING
from backend.ingestion.dataset_loader import DatasetLoader
from backend.ingestion.ground_segmentation import GroundSegmenter
from backend.mapping.cell_statistics import compute_cell_statistics
from backend.mapping.costmap import CostmapEvaluator
from backend.mapping.quadtree import AdaptiveQuadtree
from backend.tracking.clustering import EuclideanClusterer
from backend.tracking.kalman_tracker import KalmanTracker
from backend.tracking.trajectory_rollout import TrajectoryRollout
from backend.utils.synthetic_generator import SyntheticLiDARGenerator
from backend.main import PerceptionPipeline


def test_synthetic_generator():
    """Verify synthetic generator yields point clouds conforming to BOUNDS."""
    gen = SyntheticLiDARGenerator()
    points, gt = gen.generate_frame(timestamp=0.0, dt=0.04)

    assert points.ndim == 2
    assert points.shape[1] == 4  # [x, y, z, intensity]
    assert len(points) > 1000
    assert len(gt) == 2  # pedestrian and vehicle

    # Check bounds
    assert np.all(points[:, 0] >= BOUNDS.X_MIN)
    assert np.all(points[:, 0] <= BOUNDS.X_MAX)
    assert np.all(points[:, 1] >= BOUNDS.Y_MIN)
    assert np.all(points[:, 1] <= BOUNDS.Y_MAX)
    assert np.all(points[:, 2] >= BOUNDS.Z_MIN)
    assert np.all(points[:, 2] <= BOUNDS.Z_MAX)


def test_ground_segmentation():
    """Verify ground segmenter splits points into ground and obstacles accurately."""
    gen = SyntheticLiDARGenerator()
    points, _ = gen.generate_frame(timestamp=0.0, dt=0.04)

    segmenter = GroundSegmenter()
    ground_pts, obstacle_pts, mask = segmenter.segment(points)

    assert len(ground_pts) + len(obstacle_pts) == len(points)
    assert len(ground_pts) > len(obstacle_pts)  # Ground should comprise majority of points
    assert len(obstacle_pts) > 50  # Static barriers + moving obstacles detected


def test_cell_statistics():
    """Verify 2.5D cell stats calculation."""
    # Flat horizontal patch
    flat_pts = np.array([
        [0.0, 0.0, -1.6],
        [0.1, 0.0, -1.6],
        [0.0, 0.1, -1.6],
        [0.1, 0.1, -1.6],
    ])
    stats = compute_cell_statistics(flat_pts)
    assert stats is not None
    assert np.isclose(stats.delta_z, 0.0)
    assert np.isclose(stats.variance, 0.0)
    assert stats.slope < 1.0

    # Step obstacle patch
    step_pts = np.array([
        [0.0, 0.0, -1.6],
        [0.1, 0.0, -1.6],
        [0.0, 0.1, -1.3],  # 30 cm step
        [0.1, 0.1, -1.3],
    ])
    stats_step = compute_cell_statistics(step_pts)
    assert stats_step is not None
    assert np.isclose(stats_step.delta_z, 0.3)
    assert stats_step.variance > 0.01


def test_adaptive_quadtree_subdivision():
    """Verify adaptive quadtree subdivides rough areas while keeping flat areas coarse."""
    # Flat coarse cell centered at (0.25, 0.25)
    flat_x = np.random.uniform(0.1, 0.4, 50)
    flat_y = np.random.uniform(0.1, 0.4, 50)
    flat_z = np.full(50, -1.6)

    # Rough cell with step height strictly within coarse cell [5.0, 5.5] x [5.0, 5.5]
    rough_x = np.random.uniform(5.1, 5.4, 50)
    rough_y = np.random.uniform(5.1, 5.4, 50)
    rough_z = np.where(rough_x > 5.25, -1.35, -1.6)  # 25cm step inside the cell

    pts = np.column_stack([
        np.concatenate([flat_x, rough_x]),
        np.concatenate([flat_y, rough_y]),
        np.concatenate([flat_z, rough_z]),
        np.ones(100),
    ])

    qt = AdaptiveQuadtree()
    leaves = qt.build(pts)

    tree_stats = qt.get_statistics()
    assert tree_stats["total_leaves"] >= 2
    # Rough region must have subdivided into fine cells
    assert tree_stats["fine_cells"] >= 4


def test_costmap_evaluation():
    """Verify traversability cost values fall in expected bounds."""
    evaluator = CostmapEvaluator()
    qt = AdaptiveQuadtree()

    # Create flat points
    pts = np.array([
        [0.0, 0.0, -1.6, 1.0],
        [0.1, 0.1, -1.6, 1.0],
        [0.2, 0.2, -1.6, 1.0],
        [0.3, 0.3, -1.6, 1.0],
    ])
    leaves = qt.build(pts)
    evaluator.evaluate_leaves(leaves)
    for leaf in leaves:
        assert leaf.cost <= COSTMAP.SAFE_MAX

    # Add lethal obstacle point
    evaluator.apply_obstacle_occupancy(leaves, np.array([[0.1, 0.1, -0.5, 1.0]]))
    assert any(leaf.cost == COSTMAP.LETHAL_VAL for leaf in leaves)


def test_clustering_and_kalman_tracking():
    """Verify Euclidean clustering and Kalman tracker identify moving dynamic actors."""
    gen = SyntheticLiDARGenerator()
    segmenter = GroundSegmenter()
    clusterer = EuclideanClusterer()
    tracker = KalmanTracker()

    # Run for 10 frames to establish confirmed tracks
    confirmed_tracks = []
    for f in range(10):
        pts, _ = gen.generate_frame(timestamp=f * 0.04, dt=0.04)
        _, obs_pts, _ = segmenter.segment(pts)
        clusters = clusterer.cluster(obs_pts)
        confirmed_tracks = tracker.update(clusters, dt=0.04)

    assert len(confirmed_tracks) >= 1
    # Check that at least one track is dynamic (has speed > 0.25 m/s)
    dyn_tracks = tracker.get_dynamic_tracks()
    assert len(dyn_tracks) >= 1
    for dyn in dyn_tracks:
        assert dyn.speed > TRACKING.DYNAMIC_SPEED_THRESHOLD
        assert -np.pi <= dyn.heading <= np.pi


def test_trajectory_rollout():
    """Verify trajectory rollout generates valid expanding ellipses."""
    gen = SyntheticLiDARGenerator()
    segmenter = GroundSegmenter()
    clusterer = EuclideanClusterer()
    tracker = KalmanTracker()
    rollout = TrajectoryRollout()

    for f in range(8):
        pts, _ = gen.generate_frame(timestamp=f * 0.04, dt=0.04)
        _, obs_pts, _ = segmenter.segment(pts)
        clusters = clusterer.cluster(obs_pts)
        active_tracks = tracker.update(clusters, dt=0.04)

    dynamic_tracks = tracker.get_dynamic_tracks()
    assert len(dynamic_tracks) >= 1
    hazard_cones = rollout.rollout_all(dynamic_tracks)

    for tid, cones in hazard_cones.items():
        assert len(cones) == 3  # 1.0s, 2.0s, 3.0s horizons
        # Ellipse should expand with time: cone[2].semi_major > cone[0].semi_major
        assert cones[2].semi_major > cones[0].semi_major
        assert len(cones[0].polygon) == 16


def test_realtime_pipeline_benchmark():
    """Verify end-to-end perception loop latency satisfies < 35 ms (< 25 Hz target)."""
    pipeline = PerceptionPipeline(port=8769, target_fps=25.0)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Run 25 frames benchmark unthrottled to measure raw latency
    loop.run_until_complete(pipeline.run(max_frames=25, pacing=False))
    loop.close()

    assert pipeline.frame_count == 25
    # Verification of real-time target
    print(f"\n[BENCHMARK RESULT] Achieved Latency: {pipeline.rolling_latency_ms:.2f} ms | Unthrottled FPS: {pipeline.rolling_fps:.1f} Hz")
    assert pipeline.rolling_latency_ms < 35.0, f"Latency {pipeline.rolling_latency_ms}ms exceeded 35ms!"
