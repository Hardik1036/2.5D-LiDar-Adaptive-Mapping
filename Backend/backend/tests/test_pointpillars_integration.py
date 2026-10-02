"""
Integration tests for PointPillars 3D Perception in PS 26053.
Validates:
1. Integrated loading through MLPerceptionAdapter and PerceptionPipeline
2. Detections from valid LiDAR input
3. Explicit unavailable state handling (zero silent fallback)
4. Strict range enforcement ([-51.2, 51.2]m X/Y, [-5.0, 3.0]m Z) without warping
5. Points beyond +51.2m remain available to geometric path
6. Detector provenance preservation (source='pointpillars', source='geometric')
7. Conversion to DetectedCluster with velocities
8. KalmanTracker compatibility and track state updates
9. Coexistence of geometric and neural detectors
10. Determinism, error handling, and 000000.bin reference-pipeline equivalence
"""

import os
from pathlib import Path
import numpy as np
import pytest

from backend.adapters.ml_adapter import MLPerceptionAdapter
from backend.adapters.pointpillars_runtime import PointPillarsRuntimeAdapter
from backend.config import POINT_PILLARS, PointPillarsConfig
from backend.main import PerceptionPipeline
from backend.tracking.clustering import DetectedCluster, EuclideanClusterer
from backend.tracking.kalman_tracker import KalmanTracker


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "dynamic_corridor" / "velodyne"
REF_BIN = DATA_DIR / "000000.bin"


def _load_sample_sweep():
    assert REF_BIN.exists(), f"Reference sample {REF_BIN} must exist"
    raw = np.fromfile(str(REF_BIN), dtype=np.float32)
    return raw.reshape(-1, 4)


def test_1_pointpillars_adapter_loads_through_integrated_path():
    """1. PointPillars adapter loads through the integrated PS path."""
    adapter = MLPerceptionAdapter(enable_pointpillars=True)
    assert adapter.pointpillars_enabled is True
    assert adapter.pointpillars_adapter is not None
    assert adapter.pointpillars_adapter.is_available is True
    assert adapter.pointpillars_session is not None

    status = adapter.get_model_status()
    assert status["pointpillars_status"] == "AVAILABLE"
    assert status["3d_object_detection"]["status"] == "LOADED"
    assert status["3d_detection_backend"] == "POINTPILLARS_WITH_GEOMETRIC_FALLBACK"


def test_2_pointpillars_produces_detections_from_valid_input():
    """2. PointPillars produces detections from valid LiDAR input."""
    adapter = MLPerceptionAdapter(enable_pointpillars=True)
    points = _load_sample_sweep()

    res = adapter.detect_pointpillars(points)
    assert res.status == "SUCCESS"
    assert len(res.detections) > 0
    assert res.stats["pointpillars_valid_points"] > 0
    assert res.stats["active_pillars"] > 0


def test_3_pointpillars_unavailable_state_is_explicit():
    """3. PointPillars unavailable state is explicit."""
    # When disabled explicitly
    adapter_disabled = MLPerceptionAdapter(enable_pointpillars=False)
    assert adapter_disabled.pointpillars_enabled is False
    res_disabled = adapter_disabled.detect_pointpillars(np.zeros((10, 4), dtype=np.float32))
    assert res_disabled.status == "DISABLED"
    assert len(res_disabled.detections) == 0

    status_disabled = adapter_disabled.get_model_status()
    assert status_disabled["pointpillars_status"] == "NOT_AVAILABLE"
    assert status_disabled["3d_detection_backend"] == "GEOMETRIC_FALLBACK"

    # When nonexistent model path is passed
    adapter_missing = MLPerceptionAdapter(
        enable_pointpillars=True,
        detection_model_path="nonexistent_model.onnx",
    )
    assert adapter_missing.pointpillars_adapter is not None
    assert adapter_missing.pointpillars_adapter.is_available is False
    res_missing = adapter_missing.detect_pointpillars(np.zeros((10, 4), dtype=np.float32))
    assert res_missing.status == "DETECTOR_UNAVAILABLE"
    assert res_missing.error_message is not None


def test_4_no_silent_dbscan_as_pointpillars_fallback():
    """4. No silent DBSCAN-as-PointPillars fallback."""
    pipeline = PerceptionPipeline(enable_pointpillars=False)
    points = _load_sample_sweep()

    res = pipeline.process_frame(points)
    system_stats = res["system_stats"]
    assert system_stats["pointpillars_detections_count"] == 0

    # Ensure no track or cluster claims source="pointpillars" when pointpillars is disabled
    for t in res["tracks"]:
        assert getattr(t, "detector_source", "geometric") != "pointpillars"


def test_5_pointpillars_receives_only_points_inside_supported_range():
    """5. PointPillars receives only points inside its supported range."""
    adapter = MLPerceptionAdapter(enable_pointpillars=True)
    rt = adapter.pointpillars_adapter

    # Synthetic point cloud with points inside and outside range
    pts = np.array([
        [0.0, 0.0, 0.0, 0.5],          # Inside
        [50.0, 50.0, 2.0, 0.5],        # Inside
        [-50.0, -50.0, -4.0, 0.5],     # Inside
        [52.0, 0.0, 0.0, 0.5],         # Outside X > 51.2
        [-52.0, 0.0, 0.0, 0.5],        # Outside X < -51.2
        [0.0, 55.0, 0.0, 0.5],         # Outside Y > 51.2
        [0.0, 0.0, 4.0, 0.5],          # Outside Z > 3.0
        [0.0, 0.0, -6.0, 0.5],         # Outside Z < -5.0
    ], dtype=np.float32)

    res = rt.predict(pts)
    assert res.stats["input_points"] == 8
    assert res.stats["pointpillars_valid_points"] == 3
    assert res.stats["pointpillars_discarded_points"] == 5


def test_6_points_beyond_range_not_artificially_transformed():
    """6. Points beyond +51.2 m are NOT artificially transformed into PointPillars range."""
    adapter = MLPerceptionAdapter(enable_pointpillars=True)
    rt = adapter.pointpillars_adapter

    # Points at 60m, 80m, 100m
    far_pts = np.array([
        [60.0, 0.0, 0.0, 0.5],
        [80.0, 5.0, 0.0, 0.5],
        [100.0, -5.0, 0.0, 0.5],
    ], dtype=np.float32)

    res = rt.predict(far_pts)
    assert res.stats["pointpillars_valid_points"] == 0
    assert res.stats["pointpillars_discarded_points"] == 3
    assert len(res.detections) == 0


def test_7_outside_range_points_remain_available_to_geometric_path():
    """7. Outside-range points remain available to the existing geometric path."""
    from backend.tracking.object_detector import ObjectDetector3D

    clusterer = EuclideanClusterer()
    detector = ObjectDetector3D()

    # Create obstacle cluster at x = 70.0 m (outside PointPillars range, inside PS 100m range)
    far_cluster_pts = np.array([
        [70.0 + dx, 2.0 + dy, 0.5 + dz]
        for dx in np.linspace(-0.2, 0.2, 3)
        for dy in np.linspace(-0.2, 0.2, 3)
        for dz in np.linspace(-0.2, 0.2, 3)
    ], dtype=np.float32)  # 27 points

    clusters = clusterer.cluster(far_cluster_pts)
    assert len(clusters) > 0
    c = clusters[0]
    assert c.centroid[0] > 60.0
    assert c.source == "geometric"

    det_clusters = detector.detect_geometric_fallback(far_cluster_pts)
    assert len(det_clusters) > 0
    assert det_clusters[0].source == "geometric"


def test_8_detection_provenance_is_preserved():
    """8. Detection provenance is preserved."""
    # PointPillars cluster
    c_pp = DetectedCluster(
        centroid=(10.0, 2.0, 0.0),
        dimensions=(4.5, 2.0, 1.6),
        bbox=(7.75, 12.25, 1.0, 3.0, -0.8, 0.8),
        point_count=50,
        points=np.empty((0, 3), dtype=np.float32),
        source="pointpillars",
    )
    assert c_pp.source == "pointpillars"
    assert c_pp.to_dict()["source"] == "pointpillars"

    # Geometric cluster
    c_geo = DetectedCluster(
        centroid=(20.0, 5.0, 0.0),
        dimensions=(1.0, 1.0, 1.5),
        bbox=(19.5, 20.5, 4.5, 5.5, -0.75, 0.75),
        point_count=30,
        points=np.empty((0, 3), dtype=np.float32),
        source="geometric",
    )
    assert c_geo.source == "geometric"
    assert c_geo.to_dict()["source"] == "geometric"


def test_9_pointpillars_detections_convert_correctly():
    """9. PointPillars detections convert correctly to existing detection type."""
    adapter = MLPerceptionAdapter(enable_pointpillars=True)
    points = _load_sample_sweep()

    res = adapter.detect_pointpillars(points)
    clusters = [d.to_detected_cluster() for d in res.detections]

    assert len(clusters) == len(res.detections)
    for c, d in zip(clusters, res.detections):
        assert isinstance(c, DetectedCluster)
        assert c.source == "pointpillars"
        assert c.label == d.class_name
        assert c.centroid == (float(d.x), float(d.y), float(d.z))
        assert c.dimensions == (float(d.length), float(d.width), float(d.height))
        assert c.velocity == (float(d.velocity_x), float(d.velocity_y))


def test_10_tracking_accepts_pointpillars_detections():
    """10. Tracking accepts PointPillars detections."""
    tracker = KalmanTracker(max_tracks=40)

    # Ingest 3 PointPillars clusters
    clusters = [
        DetectedCluster(
            centroid=(10.0, 2.0, -0.5),
            dimensions=(4.5, 2.0, 1.6),
            bbox=(7.75, 12.25, 1.0, 3.0, -1.3, 0.3),
            point_count=50,
            points=np.empty((0, 3), dtype=np.float32),
            velocity=(2.5, 0.1),
            label="car",
            source="pointpillars",
        ),
        DetectedCluster(
            centroid=(25.0, -4.0, 0.0),
            dimensions=(5.0, 2.2, 1.8),
            bbox=(22.5, 27.5, -5.1, -2.9, -0.9, 0.9),
            point_count=40,
            points=np.empty((0, 3), dtype=np.float32),
            velocity=(0.0, 0.0),
            label="truck",
            source="pointpillars",
        ),
    ]

    active_tracks = tracker.update(clusters, dt=0.04)
    # Tentative tracks spawned
    assert len(tracker.tracks) == 2
    t0 = tracker.tracks[0]
    assert t0.label == "car"
    assert t0.vx == pytest.approx(2.5, abs=0.5)


def test_11_existing_geometric_detector_still_works():
    """11. Existing DBSCAN/geometric detector still works."""
    clusterer = EuclideanClusterer()
    rng = np.random.RandomState(42)
    pts = rng.normal(loc=[10.0, 2.0, 0.0], scale=0.1, size=(50, 3)).astype(np.float32)

    clusters = clusterer.cluster(pts)
    assert len(clusters) == 1
    assert clusters[0].source == "geometric"
    assert clusters[0].centroid[0] == pytest.approx(10.0, abs=0.2)


def test_12_existing_detector_unchanged_when_pointpillars_unavailable():
    """12. Existing detector behaviour remains unchanged when PointPillars is unavailable."""
    pipeline_disabled = PerceptionPipeline(enable_pointpillars=False)
    sample_pts = _load_sample_sweep()

    res = pipeline_disabled.process_frame(sample_pts)
    assert res["frame_id"] == 1
    assert len(res["leaves"]) > 0
    assert res["system_stats"]["pointpillars_detections_count"] == 0
    assert res["system_stats"]["pointpillars_stats"]["status"] in ("DISABLED", "NOT_AVAILABLE")


def test_13_empty_point_cloud_does_not_crash():
    """13. Empty point cloud does not crash."""
    adapter = MLPerceptionAdapter(enable_pointpillars=True)
    res = adapter.detect_pointpillars(np.empty((0, 4), dtype=np.float32))
    assert res.status == "SUCCESS"
    assert len(res.detections) == 0
    assert res.stats["input_points"] == 0
    assert res.stats["pointpillars_valid_points"] == 0


def test_14_invalid_point_shape_does_not_crash_silently():
    """14. Invalid point shape does not crash silently."""
    adapter = MLPerceptionAdapter(enable_pointpillars=True)

    with pytest.raises(ValueError):
        # 1D array
        adapter.detect_pointpillars(np.zeros(10, dtype=np.float32))

    with pytest.raises(ValueError):
        # Missing intensity channel (N, 3)
        adapter.detect_pointpillars(np.zeros((10, 3), dtype=np.float32))


def test_15_deterministic_output_for_identical_input():
    """15. Deterministic output for identical input."""
    adapter = MLPerceptionAdapter(enable_pointpillars=True)
    points = _load_sample_sweep()

    res1 = adapter.detect_pointpillars(points)
    res2 = adapter.detect_pointpillars(points)

    assert len(res1.detections) == len(res2.detections)
    for d1, d2 in zip(res1.detections, res2.detections):
        assert d1.class_name == d2.class_name
        assert d1.x == pytest.approx(d2.x, abs=1e-5)
        assert d1.y == pytest.approx(d2.y, abs=1e-5)
        assert d1.z == pytest.approx(d2.z, abs=1e-5)
        assert d1.score == pytest.approx(d2.score, abs=1e-5)


def test_16_full_ps_pipeline_processes_frame_with_pointpillars():
    """16. Full PS pipeline can process a frame with PointPillars enabled."""
    pipeline = PerceptionPipeline(enable_pointpillars=True)
    points = _load_sample_sweep()

    res = pipeline.process_frame(points)
    assert res["frame_id"] == 1
    assert len(res["leaves"]) > 0

    stats = res["system_stats"]
    assert stats["pointpillars_detections_count"] > 0
    assert stats["pointpillars_latency_ms"] > 0
    assert stats["pointpillars_stats"]["status"] == "SUCCESS"


def test_17_reference_000000_bin_equivalence():
    """
    17. 000000.bin reference test produces 84 PointPillars detections,
    subject to the tested OpenPCDet reference pipeline conditions.
    (Note: describes reference-pipeline equivalence, not real-world ground truth).
    """
    adapter = MLPerceptionAdapter(enable_pointpillars=True)
    points = _load_sample_sweep()

    res = adapter.detect_pointpillars(points)
    assert res.stats["input_points"] == 61160
    assert res.stats["pointpillars_valid_points"] == 60656
    assert res.stats["pointpillars_discarded_points"] == 504
    assert res.stats["active_pillars"] == 8628
    assert res.stats["final_detections"] == 84
    assert len(res.detections) == 84

    # Verify class breakdown matching verified Stage-1 reference
    classes = [d.class_name for d in res.detections]
    counts = {c: classes.count(c) for c in set(classes)}
    assert counts.get("car") == 72
    assert counts.get("bus") == 7
    assert counts.get("truck") == 2
    assert counts.get("motorcycle") == 1
    assert counts.get("barrier") == 1
    assert counts.get("bicycle") == 1
