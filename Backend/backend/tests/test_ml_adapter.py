"""
Unit tests for MLPerceptionAdapter and ThreatNet1D thin hazard inference.
"""

import numpy as np
import pytest

from backend.adapters.ml_adapter import MLPerceptionAdapter, evaluate_thin_hazards
from backend.ingestion.ml_adapter import evaluate_thin_hazards as evaluate_thin_hazards_ingestion
from backend.tracking.clustering import DetectedCluster


def test_process_semantic_labels():
    adapter = MLPerceptionAdapter()

    # Create 100 points
    points = np.random.uniform(-10.0, 10.0, size=(100, 3)).astype(np.float32)
    labels = np.zeros(100, dtype=np.int32)
    labels[0:40] = 0   # Ground
    labels[40:60] = 1  # Vegetation
    labels[60:85] = 2  # Rigid
    labels[85:100] = 3 # Dynamic

    res = adapter.process_semantic_labels(points, labels)

    assert len(res["ground"]) == 40
    assert len(res["vegetation"]) == 20
    assert len(res["rigid"]) == 25
    assert len(res["dynamic"]) == 15
    assert len(res["obstacle"]) == 40  # 25 + 15

    # Check empty input handling
    empty_res = adapter.process_semantic_labels(np.empty((0, 3)), np.array([]))
    assert len(empty_res["ground"]) == 0
    assert len(empty_res["obstacle"]) == 0


def test_process_pillar_detections():
    adapter = MLPerceptionAdapter()

    # [x, y, z, l, w, h, yaw, vx, vy]
    dets = np.array([
        [5.0, 3.0, 0.5, 4.0, 2.0, 1.5, 0.2, 1.2, -0.5],
        [-2.0, -4.0, 0.0, 2.5, 1.0, 1.8, -0.1, 0.0, 0.0],
    ], dtype=np.float32)

    clusters = adapter.process_pillar_detections(dets)

    assert len(clusters) == 2
    c0 = clusters[0]
    assert isinstance(c0, DetectedCluster)
    assert c0.centroid == (5.0, 3.0, 0.5)
    assert c0.dimensions == (4.0, 2.0, 1.5)
    assert c0.bbox == (3.0, 7.0, 2.0, 4.0, -0.25, 1.25)
    assert hasattr(c0, "velocity")
    assert getattr(c0, "velocity") == (pytest.approx(1.2), pytest.approx(-0.5))

    # Test empty detections
    assert adapter.process_pillar_detections(np.empty((0, 9))) == []


def test_evaluate_thin_hazards():
    """Verifies that evaluate_thin_hazards executes ThreatNet1D and flags synthetic spike strip inputs > 0.75."""
    adapter = MLPerceptionAdapter()

    # 1. Nominal terrain ground points
    gx, gy = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
    n_g = gx.size
    ground = np.column_stack([
        gx.flatten(),
        gy.flatten(),
        np.zeros(n_g, dtype=np.float32),
        np.full(n_g, 0.2, dtype=np.float32),
    ])

    # 2. Inject synthetic spike strip: delta_z = 0.035m (3.5 cm), metallic reflectance = 0.95
    sx, sy = np.meshgrid(np.linspace(1.0, 1.2, 3), np.linspace(0.0, 0.5, 4))
    n_s = sx.size
    spikes = np.column_stack([
        sx.flatten(),
        sy.flatten(),
        np.full(n_s, 0.035, dtype=np.float32),
        np.full(n_s, 0.95, dtype=np.float32),
    ])

    all_pts = np.vstack([ground, spikes])

    # Test instance method
    res = adapter.evaluate_thin_hazards(all_pts)
    assert "threat_mask" in res
    assert "probabilities" in res
    assert "threat_points" in res
    assert "threshold" in res
    assert pytest.approx(res["threshold"], abs=1e-3) == 0.75

    max_prob = float(np.max(res["probabilities"]))
    assert max_prob > 0.75, f"Spike strip probability {max_prob:.4f} must exceed threshold (0.75)"
    assert np.any(res["threat_mask"])
    assert len(res["threat_points"]) > 0

    # Test ingestion re-export function
    res_ingest = evaluate_thin_hazards_ingestion(all_pts)
    assert np.array_equal(res["threat_mask"], res_ingest["threat_mask"])


def test_evaluate_thin_hazards_empty():
    """Verifies that evaluate_thin_hazards gracefully handles empty arrays."""
    res = evaluate_thin_hazards(np.empty((0, 4)))
    assert len(res["threat_mask"]) == 0
    assert len(res["probabilities"]) == 0
    assert len(res["threat_points"]) == 0


def test_segment_point_cloud():
    """Verifies Model 1 point cloud segmentation produces valid classes."""
    adapter = MLPerceptionAdapter()

    # Empty array
    assert len(adapter.segment_point_cloud(np.empty((0, 4)))) == 0

    # Non-empty points
    pts = np.array([
        [0.0, 0.0, -1.6, 0.2],   # ground
        [1.0, 2.0, 0.5, 0.4],    # vegetation / low obstacle
        [2.0, 1.0, 1.5, 0.8],    # rigid obstacle
    ], dtype=np.float32)

    labels = adapter.segment_point_cloud(pts)
    assert len(labels) == 3
    assert labels.dtype == np.int32
    assert all(l in (0, 1, 2, 3) for l in labels)


def test_detect_low_profile_threats():
    """Verifies Model 2 ThreatNet1D low-profile hazard detection."""
    adapter = MLPerceptionAdapter()

    # Empty
    assert len(adapter.detect_low_profile_threats(np.array([]), np.array([]))) == 0

    # Nominal ground
    residuals = np.array([0.002, 0.005, 0.035], dtype=np.float32)
    intensities = np.array([0.15, 0.20, 0.95], dtype=np.float32)

    threat_mask = adapter.detect_low_profile_threats(residuals, intensities)
    assert len(threat_mask) == 3
    assert threat_mask.dtype == bool
    # Synthetic spike strip candidate (3.5 cm elevation, 0.95 intensity) should trigger
    assert threat_mask[2] == True


def test_dynamic_path_resolution():
    """Verifies dynamic model path resolution from 'models' or 'backend/models'."""
    adapter_models = MLPerceptionAdapter(model_dir="models")
    adapter_backend = MLPerceptionAdapter(model_dir="backend/models")

    assert adapter_models.threat_threshold == pytest.approx(0.75, abs=1e-3)
    assert adapter_backend.threat_threshold == pytest.approx(0.75, abs=1e-3)
    assert "0" in adapter_models.class_map or 0 in adapter_models.class_map


def test_threatnet_input_formatting():
    """Verifies that _format_threatnet_input generates buffers strictly conforming to (1, 2, 512)."""
    from backend.adapters.ml_adapter import _format_threatnet_input

    residuals = np.array([0.01, 0.02, 0.035, 0.04], dtype=np.float32)
    intensities = np.array([0.2, 0.5, 0.95, 0.8], dtype=np.float32)

    buffers, slot_mappings = _format_threatnet_input(residuals, intensities)
    assert len(buffers) == 1
    assert buffers[0].shape == (1, 2, 512)
    assert buffers[0].dtype == np.float32
    assert len(slot_mappings[0]) == 4
    for buf_pos, slot_size in slot_mappings[0]:
        assert slot_size == 5
        assert 0 <= buf_pos <= 512 - slot_size

    # Check empty input handling
    empty_bufs, empty_maps = _format_threatnet_input(np.empty(0), np.empty(0))
    assert len(empty_bufs) == 0
    assert len(empty_maps) == 0


def test_segmentation_model_instantiation():
    """Verifies that segmentation model loader instantiates SalsaNext architecture rather than raw dict."""
    adapter = MLPerceptionAdapter()
    # Ensure segmentation_model is either None or an instantiated nn.Module, never a raw dict
    assert not isinstance(adapter.segmentation_model, dict), "segmentation_model must not be a raw dict"


def test_pointpillars_config_parsing():
    """Verifies PointPillarsConfig values extracted from cbgs_pp_multihead.yaml."""
    from backend.config import POINT_PILLARS, PointPillarsConfig

    assert isinstance(POINT_PILLARS, PointPillarsConfig)
    # Spatial bounds: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    assert POINT_PILLARS.POINT_CLOUD_RANGE == (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
    assert POINT_PILLARS.x_min == -51.2
    assert POINT_PILLARS.x_max == 51.2
    assert POINT_PILLARS.y_min == -51.2
    assert POINT_PILLARS.y_max == 51.2
    assert POINT_PILLARS.z_min == -5.0
    assert POINT_PILLARS.z_max == 3.0

    # Voxel size: [0.2, 0.2, 8.0]
    assert POINT_PILLARS.VOXEL_SIZE == (0.2, 0.2, 8.0)

    # 10 Class names
    expected_classes = (
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
    assert POINT_PILLARS.CLASS_NAMES == expected_classes
    assert len(POINT_PILLARS.CLASS_NAMES) == 10

    # BEV Grid size: 512 x 512 x 1
    assert POINT_PILLARS.grid_size == (512, 512, 1)
    assert POINT_PILLARS.CODE_SIZE == 9


def test_process_pillar_detections_with_pointpillars_config():
    """Verifies MLPerceptionAdapter wiring and (M, 9) bounding box spatial filtering with PointPillarsConfig."""
    adapter = MLPerceptionAdapter()

    assert adapter.pointpillars_config is not None
    assert adapter.point_cloud_range == (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
    assert adapter.voxel_size == (0.2, 0.2, 8.0)
    assert len(adapter.pointpillars_classes) == 10
    assert adapter.box_code_size == 9

    # Create detections: 2 inside range, 2 outside range
    dets = np.array([
        # In-bounds: x=10, y=15, z=-1.0
        [10.0, 15.0, -1.0, 4.5, 2.0, 1.6, 0.5, 2.0, 0.1],
        # In-bounds: x=-20, y=-30, z=0.5
        [-20.0, -30.0, 0.5, 6.0, 2.5, 2.8, -0.3, 0.0, 0.0],
        # Out-of-bounds: x=60.0 > 51.2
        [60.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 1.0, 0.0],
        # Out-of-bounds: z=-6.0 < -5.0
        [0.0, 0.0, -6.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)

    # With filter_out_of_bounds=True (default), out-of-bounds boxes are filtered out
    filtered_clusters = adapter.process_pillar_detections(dets, filter_out_of_bounds=True)
    assert len(filtered_clusters) == 2
    assert filtered_clusters[0].centroid == (10.0, 15.0, -1.0)
    assert filtered_clusters[1].centroid == (-20.0, -30.0, 0.5)

    # With filter_out_of_bounds=False, all 4 are kept
    all_clusters = adapter.process_pillar_detections(dets, filter_out_of_bounds=False)
    assert len(all_clusters) == 4


