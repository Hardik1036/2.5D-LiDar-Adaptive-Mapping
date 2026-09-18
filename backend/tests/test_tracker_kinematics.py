"""
Unit and integration tests for 2D Non-Maximum Suppression (NMS) and Kalman Tracker kinematics.
Validates Model 2 (PointPillars) offline proposal ingestion (M, 9) and 2D EKF dynamic tracking.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pytest

from backend.config import POINT_PILLARS, TRACKING, PointPillarsConfig
from backend.tracking.clustering import DetectedCluster
from backend.tracking.kalman_tracker import KalmanTracker, SingleObjectKalmanFilter, TrackedObject
from backend.adapters.ml_adapter import MLPerceptionAdapter, non_max_suppression_2d


def _load_proposal_data() -> np.ndarray:
    """Loads offline bounding box proposals from test data assets."""
    root = Path(__file__).resolve().parent
    candidates = [
        root / "data" / "dynamic_detections_lidar_2.npy",
        root / "data" / "dynamic_detections_lidar.npy",
        Path("backend/tests/data/dynamic_detections_lidar_2.npy"),
        Path("backend/tests/data/dynamic_detections_lidar.npy"),
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return np.load(str(c))
    raise FileNotFoundError("Could not find dynamic_detections_lidar_2.npy proposal asset.")


def test_pointpillars_config_spatial_bounds():
    """Verifies that PointPillarsConfig properly defines spatial and anchor bounds."""
    assert isinstance(POINT_PILLARS, PointPillarsConfig)
    assert POINT_PILLARS.POINT_CLOUD_RANGE == (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
    assert POINT_PILLARS.VOXEL_SIZE == (0.2, 0.2, 8.0)
    assert POINT_PILLARS.CODE_SIZE == 9
    assert len(POINT_PILLARS.CLASS_NAMES) == 10
    assert POINT_PILLARS.grid_size == (512, 512, 1)


def test_load_dynamic_detections_asset():
    """Verifies that dynamic_detections_lidar_2.npy contains (8545, 9) bounding box proposals."""
    data = _load_proposal_data()
    assert isinstance(data, np.ndarray)
    assert data.shape == (8545, 9)
    assert data.dtype == np.float32

    # Check spatial bounds conform to PointPillars range [-51.2, 51.2]
    x_min, y_min, z_min, x_max, y_max, z_max = POINT_PILLARS.POINT_CLOUD_RANGE
    # Proposals are largely within or close to range limits
    assert np.all(data[:, 0] >= x_min - 1.0) and np.all(data[:, 0] <= x_max + 1.0)
    assert np.all(data[:, 1] >= y_min - 2.0) and np.all(data[:, 1] <= y_max + 2.0)


def test_non_max_suppression_2d_filters_proposals():
    """
    Verifies that basic 2D BEV NMS filters 8,545 raw dense anchor proposals down
    to distinct non-overlapping candidate objects.
    """
    data = _load_proposal_data()
    raw_count = len(data)
    assert raw_count == 8545

    # 1. Standard IoU suppression at 0.2
    filtered_02 = non_max_suppression_2d(data, iou_threshold=0.2)
    assert len(filtered_02) < raw_count
    assert len(filtered_02) < 5000
    assert filtered_02.shape[1] == 9

    # 2. More aggressive suppression with max_output_boxes limit
    top_distinct = non_max_suppression_2d(data, iou_threshold=0.2, max_output_boxes=50)
    assert len(top_distinct) <= 50
    assert len(top_distinct) > 0

    # 3. Verify synthetic exact duplicate suppression
    synthetic_boxes = np.array([
        [10.0, 10.0, 0.0, 4.0, 2.0, 1.5, 0.0, 1.0, 0.0],
        [10.1, 10.05, 0.0, 4.0, 2.0, 1.5, 0.0, 1.0, 0.0],  # Highly overlapping duplicate
        [-20.0, -15.0, 0.5, 3.5, 1.8, 1.6, 0.5, 0.0, 2.0], # Distinct object
    ], dtype=np.float32)

    synth_filtered = non_max_suppression_2d(synthetic_boxes, iou_threshold=0.3)
    assert len(synth_filtered) == 2


def test_tracker_ingestion_with_filtered_detections():
    """
    Feeds the filtered (M, 9) proposals directly into KalmanTracker
    and verifies active track states and kinematics.
    """
    data = _load_proposal_data()
    # Filter down to top 20 distinct proposals for initial frame
    filtered_proposals = non_max_suppression_2d(data, iou_threshold=0.2, max_output_boxes=20)
    assert len(filtered_proposals) > 0

    tracker = KalmanTracker(max_tracks=40)
    assert hasattr(tracker, "_lock"), "KalmanTracker must initialize a thread-safety lock"

    # Frame 1: Initial ingestion of (M, 9) array directly
    active_tracks_f1 = tracker.update(filtered_proposals, dt=0.04)
    # Tentative tracks spawned (hits = 1, not yet confirmed)
    assert len(tracker.tracks) == len(filtered_proposals)
    assert len(active_tracks_f1) == 0  # Not yet confirmed (requires MIN_HITS_TO_CONFIRM = 3)

    first_track = tracker.tracks[0]
    assert isinstance(first_track, TrackedObject)
    assert hasattr(first_track, "vx")
    assert hasattr(first_track, "vy")
    assert hasattr(first_track, "speed")
    assert hasattr(first_track, "heading")
    assert isinstance(first_track.dimensions, tuple)


def test_tracker_kinematics_multi_frame_convergence():
    """
    Simulates consecutive kinematic frames with constant velocity motion
    to verify Hungarian data association, Kalman state convergence, and track confirmation.
    """
    data = _load_proposal_data()
    # Take 5 distinct proposals with nonzero velocity
    proposals = non_max_suppression_2d(data, iou_threshold=0.2, max_output_boxes=30)
    moving_mask = np.hypot(proposals[:, 7], proposals[:, 8]) > 0.5
    moving_proposals = proposals[moving_mask]
    if len(moving_proposals) < 3:
        moving_proposals = proposals[:5]

    selected_dets = moving_proposals[:5].copy()
    num_objects = len(selected_dets)

    tracker = KalmanTracker(gating_dist=2.5, max_tracks=40)
    dt = 0.04

    # Run for 6 consecutive frames propagating coordinates via true kinematics
    current_dets = selected_dets.copy()
    confirmed_tracks = []

    for frame in range(6):
        confirmed_tracks = tracker.update(current_dets, dt=dt)
        # Advance object positions: x += vx * dt, y += vy * dt
        current_dets[:, 0] += current_dets[:, 7] * dt
        current_dets[:, 1] += current_dets[:, 8] * dt

    # After 6 frames (> MIN_HITS_TO_CONFIRM=3), tracks must be confirmed
    assert len(confirmed_tracks) >= 1
    assert len(confirmed_tracks) <= num_objects

    for trk in confirmed_tracks:
        assert trk.is_confirmed
        assert trk.hits >= 3
        assert trk.time_since_update == 0
        assert trk.speed >= 0.0
        assert -np.pi <= trk.heading <= np.pi
        # Verify bounding box is centered around state
        min_x, max_x, min_y, max_y, min_z, max_z = trk.bbox
        assert min_x < max_x
        assert min_y < max_y


def test_tracker_robustness_and_edge_cases():
    """Verifies edge case handling in KalmanTracker: empty arrays, 1D arrays, and out-of-bounds."""
    tracker = KalmanTracker()

    # 1. Empty array
    assert tracker.update(np.empty((0, 9))) == []

    # 2. Single 1D proposal array (9,)
    single_prop = np.array([5.0, -10.0, 0.0, 4.2, 1.8, 1.5, 0.1, 1.5, 0.2], dtype=np.float32)
    tracker.update(single_prop)
    assert len(tracker.tracks) == 1

    # 3. Thread-safety lock usage
    with tracker._lock:
        tracks = tracker.get_confirmed_tracks()
        assert isinstance(tracks, list)


def test_tracker_non_finite_velocity_rejection():
    """
    Verifies that non-finite velocities (NaN, +inf, -inf) are rejected during:
    1. SingleObjectKalmanFilter initialization
    2. SingleObjectKalmanFilter measurement updates
    3. KalmanTracker track spawning from DetectedCluster and ndarray proposals
    4. Hungarian data association with subsequent frames
    """
    # 1. SingleObjectKalmanFilter initialization with non-finite velocities
    kf = SingleObjectKalmanFilter(
        init_x=10.0,
        init_y=20.0,
        init_vx=float("inf"),
        init_vy=float("nan"),
    )
    assert kf.vx == 0.0
    assert kf.vy == 0.0
    assert np.isfinite(kf.px) and np.isfinite(kf.pvx)

    # 2. Measurement updates with non-finite velocities
    kf.vx = 1.0
    kf.vy = 2.0
    # Update with inf and nan velocities (and meas_x=kf.x, meas_y=kf.y so position innovation is 0):
    # non-finite velocities should be ignored, preserving existing velocities
    kf.update(kf.x, kf.y, meas_vx=float("inf"), meas_vy=float("-inf"))
    assert kf.vx == 1.0
    assert kf.vy == 2.0

    kf.update(kf.x, kf.y, meas_vx=float("nan"), meas_vy=float("nan"))
    assert kf.vx == 1.0
    assert kf.vy == 2.0

    # Finite velocity update should blend normally: 0.7 * 1.0 + 0.3 * 2.0 = 1.3
    kf.update(kf.x, kf.y, meas_vx=2.0, meas_vy=3.0)
    assert np.isclose(kf.vx, 0.7 * 1.0 + 0.3 * 2.0)
    assert np.isclose(kf.vy, 0.7 * 2.0 + 0.3 * 3.0)

    # 3. KalmanTracker track spawning with non-finite velocities in DetectedCluster
    tracker = KalmanTracker()
    bad_cluster = DetectedCluster(
        centroid=(5.0, 5.0, 0.0),
        dimensions=(1.0, 1.0, 1.0),
        bbox=(4.5, 5.5, 4.5, 5.5, -0.5, 0.5),
        point_count=30,
        points=np.empty((0, 3), dtype=np.float32),
        velocity=(float("inf"), float("nan")),
    )
    tracker.update([bad_cluster], dt=0.04)
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].vx == 0.0
    assert tracker.tracks[0].vy == 0.0

    # 4. Subsequent frame update with inf in proposal array
    # If velocity were not rejected, next predict would produce non-finite coords,
    # causing Hungarian linear_sum_assignment to raise ValueError on invalid numeric entries.
    proposal_frame2 = np.array([
        [5.1, 5.1, 0.0, 1.0, 1.0, 1.0, 0.0, float("-inf"), float("nan")]
    ], dtype=np.float32)
    tracker.update(proposal_frame2, dt=0.04)
    assert len(tracker.tracks) == 1
    assert np.isfinite(tracker.tracks[0].x)
    assert np.isfinite(tracker.tracks[0].y)
    assert np.isfinite(tracker.tracks[0].vx)
    assert np.isfinite(tracker.tracks[0].vy)


def test_tracker_pruning_before_max_tracks_enforcement():
    """
    Verifies that expired tracks are pruned before checking max_tracks,
    allowing unassigned detections to occupy the newly freed track slots.
    """
    tracker = KalmanTracker(max_tracks=3, max_age=2)
    # Frame 1: spawn 3 tracks at locations (0,0), (10,10), (20,20)
    f1 = np.array([
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [10.0, 10.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [20.0, 20.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)
    tracker.update(f1)
    assert len(tracker.tracks) == 3

    # Frame 2 & 3: Only object 0 is detected; objects 1 and 2 miss updates
    f_single = np.array([
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)
    tracker.update(f_single)
    tracker.update(f_single)

    # Frame 4: Object 0 is still detected, plus a new object at (50, 50).
    # Since missed objects 1 and 2 have exceeded max_age=2, they should be pruned
    # BEFORE enforcing max_tracks=3, allowing the new detection at (50, 50) to spawn.
    f_new = np.array([
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [50.0, 50.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)
    tracker.update(f_new)

    assert len(tracker.tracks) == 2
    # Ensure (50.0, 50.0) was successfully spawned into freed slot
    assert any(abs(t.x - 50.0) < 1.0 and abs(t.y - 50.0) < 1.0 for t in tracker.tracks)


