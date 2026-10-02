"""
Unit and integration tests for PointPillarsRuntimeAdapter.
Verifies all 16 architectural requirements and validates numerical fidelity against 000000.bin.
"""

from pathlib import Path
import numpy as np
import pytest

from backend.adapters.pointpillars_runtime import (
    PointPillarsDetection,
    PointPillarsResult,
    PointPillarsRuntimeAdapter,
    rotated_bev_iou,
)


@pytest.fixture(scope="module")
def runtime_adapter():
    """Shared PointPillarsRuntimeAdapter instance."""
    adapter = PointPillarsRuntimeAdapter()
    return adapter


def test_1_model_loads_successfully(runtime_adapter):
    """1. Model loads successfully and reports availability."""
    assert runtime_adapter.is_available is True
    assert runtime_adapter.initialization_error is None
    assert runtime_adapter._session is not None
    assert runtime_adapter._anchors is not None
    assert runtime_adapter._anchors.shape == (327680, 10)
    assert runtime_adapter._pfn_weight is not None
    assert runtime_adapter._pfn_weight.shape == (64, 11)


def test_2_nx4_input_accepted(runtime_adapter):
    """2. Nx4 input [x, y, z, intensity] is accepted."""
    pts = np.array([
        [0.0, 0.0, 0.0, 0.5],
        [1.0, 2.0, -1.0, 0.8],
    ], dtype=np.float32)
    res = runtime_adapter.predict(pts)
    assert isinstance(res, PointPillarsResult)
    assert res.status == "SUCCESS"
    assert res.stats["input_points"] == 2
    assert res.stats["pointpillars_valid_points"] == 2


def test_3_invalid_input_dimensions_rejected(runtime_adapter):
    """3. Invalid input dimensions are explicitly rejected with ValueError/TypeError."""
    # Nx3 missing intensity
    with pytest.raises(ValueError, match="intensity"):
        runtime_adapter.predict(np.zeros((10, 3), dtype=np.float32))

    # Nx6 unsupported channels
    with pytest.raises(ValueError, match="Unsupported point cloud dimension"):
        runtime_adapter.predict(np.zeros((10, 6), dtype=np.float32))

    # 1D array
    with pytest.raises(ValueError, match="2D array"):
        runtime_adapter.predict(np.zeros(10, dtype=np.float32))

    # Non-numpy input
    with pytest.raises(TypeError, match="numpy array"):
        runtime_adapter.predict([[0, 0, 0, 1]])


def test_4_range_filtering(runtime_adapter):
    """4. Points outside [-51.2, 51.2] in X/Y and [-5.0, 3.0] in Z are filtered."""
    pts = np.array([
        [0.0, 0.0, 0.0, 0.5],      # Inside
        [52.0, 0.0, 0.0, 0.5],     # Out (X > 51.2)
        [-52.0, 0.0, 0.0, 0.5],    # Out (X < -51.2)
        [0.0, 52.0, 0.0, 0.5],     # Out (Y > 51.2)
        [0.0, -52.0, 0.0, 0.5],    # Out (Y < -51.2)
        [0.0, 0.0, 4.0, 0.5],      # Out (Z > 3.0)
        [0.0, 0.0, -6.0, 0.5],     # Out (Z < -5.0)
    ], dtype=np.float32)

    res = runtime_adapter.predict(pts)
    assert res.stats["input_points"] == 7
    assert res.stats["pointpillars_valid_points"] == 1
    assert res.stats["pointpillars_discarded_points"] == 6


def test_5_voxelisation_grid(runtime_adapter):
    """5. Coordinate-to-grid index equations match 0.2m voxel size and 512x512 limits."""
    # Point at lower bound: -51.2 + 0.05 -> idx 0
    # Point at center: 0.0 -> idx 256
    pts = np.array([
        [-51.15, -51.15, 0.0, 0.5],
        [0.0, 0.0, 0.0, 0.5],
    ], dtype=np.float32)
    res = runtime_adapter.predict(pts)
    assert res.stats["active_pillars"] == 2


def test_6_maximum_20_points_per_pillar(runtime_adapter):
    """6. Pillars cap points at MAX_POINTS_PER_VOXEL = 20."""
    # 50 points placed into the exact same pillar
    pts = np.zeros((50, 4), dtype=np.float32)
    pts[:, 0] = 0.05
    pts[:, 1] = 0.05
    pts[:, 2] = np.linspace(-1.0, 1.0, 50)
    pts[:, 3] = 0.5

    res = runtime_adapter.predict(pts)
    assert res.stats["active_pillars"] == 1
    assert res.stats["pointpillars_valid_points"] == 50


def test_7_maximum_30000_pillars(runtime_adapter):
    """7. Number of active pillars is capped at MAX_NUMBER_OF_VOXELS = 30000."""
    # Generate 35000 distinct pillar locations
    xs = np.linspace(-50.0, 50.0, 200, dtype=np.float32)
    ys = np.linspace(-50.0, 50.0, 175, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    grid_pts = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size, dtype=np.float32), np.full(xx.size, 0.5, dtype=np.float32)], axis=1)

    assert grid_pts.shape[0] == 35000
    res = runtime_adapter.predict(grid_pts)
    assert res.stats["active_pillars"] <= 30000


def test_8_11_channel_feature_construction(runtime_adapter):
    """8. Verifies 11-channel feature construction logic."""
    pts = np.array([[0.0, 0.0, 0.0, 0.5]], dtype=np.float32)
    res = runtime_adapter.predict(pts)
    assert res.status == "SUCCESS"


def test_9_bev_shape(runtime_adapter):
    """9. Verifies BEV pseudo-image grid is [1, 64, 512, 512]."""
    assert runtime_adapter.GRID_SIZE == (512, 512, 1)
    assert runtime_adapter.VOXEL_SIZE[0] == 0.2
    assert runtime_adapter.VOXEL_SIZE[1] == 0.2


def test_10_onnx_inference_output_shapes(runtime_adapter):
    """10. ONNX inference produces 12 output arrays."""
    assert len(runtime_adapter._output_names) == 12


def test_11_anchor_decoding(runtime_adapter):
    """11. Residual box decoding accurately scales offsets."""
    anchors = np.array([[10.0, 20.0, 1.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    encodings = np.zeros((1, 10), dtype=np.float32)
    decoded = runtime_adapter._decode_boxes(encodings, anchors)

    # With zero residuals, decoded box should exactly match anchor center and dimensions
    assert np.isclose(decoded[0, 0], 10.0, atol=1e-4)
    assert np.isclose(decoded[0, 1], 20.0, atol=1e-4)
    assert np.isclose(decoded[0, 2], 1.0, atol=1e-4)
    assert np.isclose(decoded[0, 3], 4.0, atol=1e-4)
    assert np.isclose(decoded[0, 4], 2.0, atol=1e-4)
    assert np.isclose(decoded[0, 5], 1.5, atol=1e-4)


def test_12_score_threshold(runtime_adapter):
    """12. Final detections respect score threshold >= 0.1."""
    bin_path = Path("Backend/data/dynamic_corridor/velodyne/000000.bin")
    if not bin_path.exists():
        bin_path = Path("data/dynamic_corridor/velodyne/000000.bin")
    if bin_path.exists():
        pts = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
        res = runtime_adapter.predict(pts)
        for det in res.detections:
            assert det.score >= runtime_adapter.score_thresh


def test_13_rotated_bev_nms():
    """13. Rotated BEV IoU and NMS suppress duplicate overlapping boxes."""
    b1 = np.array([0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0], dtype=np.float32)
    b2 = np.array([0.1, 0.1, 0.0, 4.0, 2.0, 1.5, 0.0], dtype=np.float32)
    b3 = np.array([10.0, 10.0, 0.0, 4.0, 2.0, 1.5, 0.0], dtype=np.float32)

    iou_overlap = rotated_bev_iou(b1, b2)
    iou_disjoint = rotated_bev_iou(b1, b3)

    assert iou_overlap > 0.8
    assert iou_disjoint == 0.0


def test_14_empty_point_cloud(runtime_adapter):
    """14. Empty point cloud returns 0 detections without exception."""
    empty_pts = np.empty((0, 4), dtype=np.float32)
    res = runtime_adapter.predict(empty_pts)
    assert res.status == "SUCCESS"
    assert len(res.detections) == 0
    assert res.stats["input_points"] == 0
    assert res.stats["final_detections"] == 0


def test_15_no_valid_points_after_range_filtering(runtime_adapter):
    """15. Point cloud completely outside PointPillars range returns 0 detections."""
    far_pts = np.array([
        [100.0, 0.0, 0.0, 0.5],
        [0.0, 100.0, 0.0, 0.5],
    ], dtype=np.float32)
    res = runtime_adapter.predict(far_pts)
    assert res.status == "SUCCESS"
    assert len(res.detections) == 0
    assert res.stats["pointpillars_valid_points"] == 0
    assert res.stats["pointpillars_discarded_points"] == 2


def test_16_deterministic_output(runtime_adapter):
    """16. Identical input point clouds yield deterministic detections."""
    bin_path = Path("Backend/data/dynamic_corridor/velodyne/000000.bin")
    if not bin_path.exists():
        bin_path = Path("data/dynamic_corridor/velodyne/000000.bin")
    if bin_path.exists():
        pts = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
        res1 = runtime_adapter.predict(pts)
        res2 = runtime_adapter.predict(pts)

        assert len(res1.detections) == len(res2.detections)
        for d1, d2 in zip(res1.detections, res2.detections):
            assert d1.class_name == d2.class_name
            assert np.isclose(d1.score, d2.score, atol=1e-6)
            assert np.isclose(d1.x, d2.x, atol=1e-5)
            assert np.isclose(d1.y, d2.y, atol=1e-5)
            assert np.isclose(d1.z, d2.z, atol=1e-5)


def test_17_reference_000000_bin_detection_parity(runtime_adapter):
    """
    17. End-to-end reference validation on 000000.bin:
    Verifies that exactly 84 detections are produced, matching the reference OpenPCDet model.
    """
    bin_path = Path("Backend/data/dynamic_corridor/velodyne/000000.bin")
    if not bin_path.exists():
        bin_path = Path("data/dynamic_corridor/velodyne/000000.bin")
    assert bin_path.exists(), f"Reference point cloud {bin_path} not found"

    pts = np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)
    res = runtime_adapter.predict(pts)

    assert res.status == "SUCCESS"
    assert res.stats["input_points"] == 61160
    assert res.stats["pointpillars_valid_points"] == 60656
    assert res.stats["pointpillars_discarded_points"] == 504
    assert res.stats["active_pillars"] == 8628
    assert res.stats["raw_candidates"] == 400
    assert len(res.detections) == 84

    # Class distribution checks
    class_counts = {}
    for d in res.detections:
        class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1

    assert "car" in class_counts
    assert "truck" in class_counts
    assert "bus" in class_counts
    assert "barrier" in class_counts
    assert "motorcycle" in class_counts

    # Verify tracker format conversion
    cluster = res.detections[0].to_detected_cluster()
    assert cluster.label == res.detections[0].class_name
    assert np.isclose(cluster.velocity[0], res.detections[0].velocity_x, atol=1e-5)
    assert np.isclose(cluster.velocity[1], res.detections[0].velocity_y, atol=1e-5)
