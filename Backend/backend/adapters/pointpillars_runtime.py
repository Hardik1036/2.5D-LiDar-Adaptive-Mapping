"""
PointPillars 3D Perception Runtime Adapter for DRDO SIH 2026 Problem Statement 53.

Encapsulates complete standalone end-to-end inference:
1. Range filtering: X in [-51.2, 51.2], Y in [-51.2, 51.2], Z in [-5.0, 3.0]
2. 5D Point feature preparation (x, y, z, intensity, dt=0.0)
3. Voxelisation (0.2m x 0.2m x 8.0m, 512 x 512 grid, max 20 points, max 30000 pillars)
4. 11-channel PillarVFE feature construction with trained PFN weights
5. PointPillar BEV pseudo-image scatter ([1, 64, 512, 512])
6. ONNX Runtime inference of BaseBEVBackbone + shared_conv + 6-head AnchorHeadMulti
7. Anchor alignment and residual box decoding (x, y, z, l, w, h, yaw, vx, vy)
8. Multi-class rotated BEV NMS (score_thresh=0.1, nms_thresh=0.2)
9. Output formatting compatible with PS 26053 detector/tracker pipeline.

Preserves zero-fabrication rules:
- Fails explicitly if model or dependencies are unavailable; never silently falls back to DBSCAN.
- Exposes raw point and pillar metrics.
- Excludes out-of-range points strictly without synthetic translation or scaling.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Optional OpenCV import for rotated BEV IoU
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

# Optional ONNX Runtime import
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    ort = None
    HAS_ORT = False

from backend.tracking.clustering import DetectedCluster

logger = logging.getLogger("PointPillarsRuntime")

# Class names for nuScenes 10-class taxonomy
POINTPILLARS_CLASS_NAMES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"
]

# Head-to-Class index mapping (1-indexed global class ID)
HEAD_CLASS_MAPPINGS = [
    [1],        # Head 0: car
    [2, 3],     # Head 1: truck, construction_vehicle
    [4, 5],     # Head 2: bus, trailer
    [6],        # Head 3: barrier
    [7, 8],     # Head 4: motorcycle, bicycle
    [9, 10],    # Head 5: pedestrian, traffic_cone
]


@dataclass
class PointPillarsDetection:
    """Represents a 3D oriented bounding box detection produced by PointPillars."""
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    yaw: float
    velocity_x: float
    velocity_y: float
    class_name: str
    class_id: int
    score: float

    def to_dict(self) -> Dict[str, Any]:
        """Serializes detection to JSON-serializable dictionary."""
        return {
            "x": round(float(self.x), 4),
            "y": round(float(self.y), 4),
            "z": round(float(self.z), 4),
            "length": round(float(self.length), 4),
            "width": round(float(self.width), 4),
            "height": round(float(self.height), 4),
            "yaw": round(float(self.yaw), 4),
            "velocity_x": round(float(self.velocity_x), 4),
            "velocity_y": round(float(self.velocity_y), 4),
            "velocity": (round(float(self.velocity_x), 4), round(float(self.velocity_y), 4)),
            "class_name": self.class_name,
            "class_id": self.class_id,
            "score": round(float(self.score), 4),
        }

    def to_detected_cluster(self, points: Optional[np.ndarray] = None) -> DetectedCluster:
        """
        Converts PointPillars detection into PS 26053 DetectedCluster format
        for direct compatibility with tracking and Kalman filtering.
        """
        half_l = self.length / 2.0
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        bbox = (
            float(self.x - half_l), float(self.x + half_l),
            float(self.y - half_w), float(self.y + half_w),
            float(self.z - half_h), float(self.z + half_h),
        )
        pt_arr = points if points is not None else np.empty((0, 3), dtype=np.float32)
        return DetectedCluster(
            centroid=(float(self.x), float(self.y), float(self.z)),
            dimensions=(float(self.length), float(self.width), float(self.height)),
            bbox=bbox,
            point_count=len(pt_arr),
            points=pt_arr,
            yaw=float(self.yaw),
            velocity=(float(self.velocity_x), float(self.velocity_y)),
            label=self.class_name,
            source="pointpillars",
        )


@dataclass
class PointPillarsResult:
    """Execution container returning detections, detailed metrics, and runtime status."""
    detections: List[PointPillarsDetection] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    status: str = "INITIALIZED"
    error_message: Optional[str] = None


def rotated_bev_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Computes exact 2D rotated bounding box intersection-over-union (IoU) in BEV.
    Box format: [x, y, z, dx, dy, dz, yaw, ...]
    """
    if not HAS_CV2 or cv2 is None:
        # Fallback to axis-aligned BEV box IoU if OpenCV is missing
        x1_min, x1_max = box1[0] - box1[3] / 2.0, box1[0] + box1[3] / 2.0
        y1_min, y1_max = box1[1] - box1[4] / 2.0, box1[1] + box1[4] / 2.0
        x2_min, x2_max = box2[0] - box2[3] / 2.0, box2[0] + box2[3] / 2.0
        y2_min, y2_max = box2[1] - box2[4] / 2.0, box2[1] + box2[4] / 2.0
        inter_x = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
        inter_y = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
        inter = inter_x * inter_y
        union = box1[3] * box1[4] + box2[3] * box2[4] - inter
        return inter / max(union, 1e-6)

    rect1 = ((float(box1[0]), float(box1[1])), (float(box1[3]), float(box1[4])), float(np.degrees(box1[6])))
    rect2 = ((float(box2[0]), float(box2[1])), (float(box2[3]), float(box2[4])), float(np.degrees(box2[6])))

    ret, pts = cv2.rotatedRectangleIntersection(rect1, rect2)
    if ret == cv2.INTERSECT_NONE or pts is None or len(pts) < 3:
        return 0.0

    inter_area = float(cv2.contourArea(pts))
    area1 = float(box1[3] * box1[4])
    area2 = float(box2[3] * box2[4])
    union_area = area1 + area2 - inter_area
    return inter_area / max(union_area, 1e-6)


class PointPillarsRuntimeAdapter:
    """
    Standalone, zero-PyTorch-dependency PointPillars runtime inference engine.
    Runs full preprocessing, ONNX Runtime backbone/head inference, and vectorized box decoding.
    """

    POINT_CLOUD_RANGE = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], dtype=np.float32)
    VOXEL_SIZE = np.array([0.2, 0.2, 8.0], dtype=np.float32)
    GRID_SIZE = (512, 512, 1)
    MAX_POINTS_PER_VOXEL = 20
    MAX_NUMBER_OF_VOXELS = 30000

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        weights_path: Optional[Union[str, Path]] = None,
        anchors_path: Optional[Union[str, Path]] = None,
        score_thresh: float = 0.1,
        nms_thresh: float = 0.2,
        nms_pre_maxsize: int = 1000,
        nms_post_maxsize: int = 83,
    ):
        self.score_thresh = float(score_thresh)
        self.nms_thresh = float(nms_thresh)
        self.nms_pre_maxsize = nms_pre_maxsize
        self.nms_post_maxsize = nms_post_maxsize

        self.is_available = False
        self.initialization_error: Optional[str] = None

        self._session: Optional[Any] = None
        self._input_name: Optional[str] = None
        self._output_names: List[str] = []

        # PFN parameters
        self._pfn_weight: Optional[np.ndarray] = None
        self._pfn_norm_weight: Optional[np.ndarray] = None
        self._pfn_norm_bias: Optional[np.ndarray] = None
        self._pfn_norm_mean: Optional[np.ndarray] = None
        self._pfn_norm_var: Optional[np.ndarray] = None
        self._pfn_norm_eps: float = 1e-3

        # Anchors (327680, 10)
        self._anchors: Optional[np.ndarray] = None

        # Resolve asset locations
        self.model_path = self._resolve_model_path(model_path)
        self.weights_path = self._resolve_weights_path(weights_path)
        self.anchors_path = self._resolve_anchors_path(anchors_path)

        # Initialize
        self._init_runtime()

    def _resolve_model_path(self, override_path: Optional[Union[str, Path]]) -> Optional[Path]:
        if override_path is not None:
            return Path(override_path)

        root = Path(__file__).resolve().parent.parent
        candidates = [
            root / "models" / "cbgs_pp_backbone_head.onnx",
            Path("backend/models/cbgs_pp_backbone_head.onnx"),
            Path("models/cbgs_pp_backbone_head.onnx"),
            root.parent.parent / "PointPillars_Assets" / "cbgs_pp_backbone_head.onnx",
            Path("PointPillars_Assets/cbgs_pp_backbone_head.onnx"),
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                return c.resolve()
        return None

    def _resolve_weights_path(self, override_path: Optional[Union[str, Path]]) -> Optional[Path]:
        if override_path is not None:
            return Path(override_path)

        root = Path(__file__).resolve().parent.parent
        candidates = [
            root / "models" / "pointpillars_pfn_weights.npz",
            Path("backend/models/pointpillars_pfn_weights.npz"),
            Path("models/pointpillars_pfn_weights.npz"),
            root.parent.parent / "PointPillars_Assets" / "pointpillars_pfn_weights.npz",
            Path("PointPillars_Assets/pointpillars_pfn_weights.npz"),
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                return c.resolve()
        return None

    def _resolve_anchors_path(self, override_path: Optional[Union[str, Path]]) -> Optional[Path]:
        if override_path is not None:
            return Path(override_path)

        root = Path(__file__).resolve().parent.parent
        candidates = [
            root / "models" / "pointpillars_anchors.npz",
            Path("backend/models/pointpillars_anchors.npz"),
            Path("models/pointpillars_anchors.npz"),
            root.parent.parent / "PointPillars_Assets" / "pointpillars_anchors.npz",
            Path("PointPillars_Assets/pointpillars_anchors.npz"),
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                return c.resolve()
        return None

    def _init_runtime(self) -> None:
        if not HAS_ORT or ort is None:
            self.initialization_error = "onnxruntime is not installed in current Python environment"
            logger.warning(f"PointPillars unavailable: {self.initialization_error}")
            return

        if self.model_path is None or not self.model_path.exists():
            self.initialization_error = "PointPillars ONNX model file cbgs_pp_backbone_head.onnx not found"
            logger.warning(f"PointPillars unavailable: {self.initialization_error}")
            return

        if self.weights_path is None or not self.weights_path.exists():
            self.initialization_error = "PointPillars PFN weights file pointpillars_pfn_weights.npz not found"
            logger.warning(f"PointPillars unavailable: {self.initialization_error}")
            return

        if self.anchors_path is None or not self.anchors_path.exists():
            self.initialization_error = "PointPillars anchors file pointpillars_anchors.npz not found"
            logger.warning(f"PointPillars unavailable: {self.initialization_error}")
            return

        try:
            # 1. Load PFN weights
            pfn_data = np.load(str(self.weights_path))
            self._pfn_weight = pfn_data["pfn_weight"]
            self._pfn_norm_weight = pfn_data["pfn_norm_weight"]
            self._pfn_norm_bias = pfn_data["pfn_norm_bias"]
            self._pfn_norm_mean = pfn_data["pfn_norm_mean"]
            self._pfn_norm_var = pfn_data["pfn_norm_var"]
            self._pfn_norm_eps = float(pfn_data.get("pfn_norm_eps", 1e-3))

            # 2. Load Anchors
            anchors_data = np.load(str(self.anchors_path))
            self._anchors = anchors_data["anchors"].astype(np.float32)

            # 3. Create ONNX Runtime Inference Session
            available_providers = ort.get_available_providers()
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available_providers else ["CPUExecutionProvider"]

            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 1
            sess_opts.inter_op_num_threads = 1
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._session = ort.InferenceSession(str(self.model_path), sess_opts, providers=providers)
            self._input_name = self._session.get_inputs()[0].name
            self._output_names = [o.name for o in self._session.get_outputs()]

            self.is_available = True
            logger.info(f"PointPillars ONNX Runtime initialized successfully from {self.model_path}")

        except Exception as e:
            self.initialization_error = f"Failed to initialize PointPillars ONNX session: {e}"
            logger.error(self.initialization_error)
            self.is_available = False

    def predict(self, points: np.ndarray) -> PointPillarsResult:
        """
        Executes complete PointPillars 3D object detection on raw LiDAR points.

        Args:
            points: (N, 4) numpy array representing [x, y, z, intensity].

        Returns:
            PointPillarsResult with detections, statistics, and execution status.
        """
        if not self.is_available or self._session is None:
            return PointPillarsResult(
                detections=[],
                stats={
                    "input_points": len(points) if isinstance(points, np.ndarray) else 0,
                    "pointpillars_valid_points": 0,
                    "pointpillars_discarded_points": len(points) if isinstance(points, np.ndarray) else 0,
                    "active_pillars": 0,
                    "raw_candidates": 0,
                    "final_detections": 0,
                },
                status="DETECTOR_UNAVAILABLE",
                error_message=self.initialization_error or "PointPillars detector is unavailable",
            )

        # 1. Input Validation
        t_start = time.perf_counter()
        t0 = t_start
        if not isinstance(points, np.ndarray):
            raise TypeError(f"PointPillars input must be a numpy array, got {type(points)}")

        if points.ndim != 2:
            raise ValueError(f"PointPillars expects a 2D array of shape (N, 4), got {points.shape}")

        if points.shape[1] == 3:
            raise ValueError(f"PointPillars requires intensity channel [x, y, z, intensity]. Got shape {points.shape}")
        elif points.shape[1] == 4:
            # Append single-sweep dt = 0.0
            dt = np.zeros((points.shape[0], 1), dtype=np.float32)
            pts_5 = np.hstack([points[:, :4].astype(np.float32), dt])
        elif points.shape[1] == 5:
            pts_5 = points.astype(np.float32)
        else:
            raise ValueError(f"Unsupported point cloud dimension: {points.shape}. Expected (N, 4) [x, y, z, intensity].")

        total_input_points = len(pts_5)

        # 2. Strict Range Filtering (Learned PointPillars training domain)
        min_b = self.POINT_CLOUD_RANGE[:3]
        max_b = self.POINT_CLOUD_RANGE[3:]
        valid_mask = (
            (pts_5[:, 0] >= min_b[0]) & (pts_5[:, 0] <= max_b[0]) &
            (pts_5[:, 1] >= min_b[1]) & (pts_5[:, 1] <= max_b[1]) &
            (pts_5[:, 2] >= min_b[2]) & (pts_5[:, 2] <= max_b[2])
        )
        valid_pts = pts_5[valid_mask]
        num_valid = len(valid_pts)
        num_discarded = total_input_points - num_valid
        t_prep_ms = (time.perf_counter() - t0) * 1000.0

        stats: Dict[str, Any] = {
            "input_points": total_input_points,
            "pointpillars_valid_points": num_valid,
            "pointpillars_discarded_points": num_discarded,
            "active_pillars": 0,
            "raw_candidates": 0,
            "final_detections": 0,
        }

        if num_valid == 0:
            stats["timings_ms"] = {
                "point_prep_ms": round(t_prep_ms, 2),
                "voxelisation_ms": 0.0,
                "pfn_scatter_ms": 0.0,
                "onnx_inference_ms": 0.0,
                "decoding_ms": 0.0,
                "nms_ms": 0.0,
                "total_time_ms": round(t_prep_ms, 2),
            }
            return PointPillarsResult(
                detections=[],
                stats=stats,
                status="SUCCESS",
                error_message=None,
            )

        # 3. Voxelisation / Pillar Indexing
        t0 = time.perf_counter()
        x_idx = np.floor((valid_pts[:, 0] - min_b[0]) / self.VOXEL_SIZE[0]).astype(np.int32)
        y_idx = np.floor((valid_pts[:, 1] - min_b[1]) / self.VOXEL_SIZE[1]).astype(np.int32)
        x_idx = np.clip(x_idx, 0, self.GRID_SIZE[0] - 1)
        y_idx = np.clip(y_idx, 0, self.GRID_SIZE[1] - 1)
        pillar_keys = y_idx * self.GRID_SIZE[0] + x_idx

        unique_keys, inverse_idx, counts = np.unique(pillar_keys, return_inverse=True, return_counts=True)
        M = min(len(unique_keys), self.MAX_NUMBER_OF_VOXELS)
        stats["active_pillars"] = M

        # Build [M, 20, 5] voxels and coordinates
        voxels = np.zeros((M, self.MAX_POINTS_PER_VOXEL, 5), dtype=np.float32)
        voxel_num_points = np.zeros(M, dtype=np.int32)
        coords = np.zeros((M, 4), dtype=np.int32)
        coords[:, 2] = unique_keys[:M] // self.GRID_SIZE[0]  # y_idx
        coords[:, 3] = unique_keys[:M] % self.GRID_SIZE[0]   # x_idx

        order = np.argsort(inverse_idx)
        sorted_pts = valid_pts[order]
        splits = np.split(sorted_pts, np.cumsum(counts)[:-1])

        for i in range(M):
            chunk = splits[i]
            n = min(len(chunk), self.MAX_POINTS_PER_VOXEL)
            voxels[i, :n] = chunk[:n]
            voxel_num_points[i] = n
        t_voxel_ms = (time.perf_counter() - t0) * 1000.0

        # 4. Exact 11-Channel PillarVFE Feature Construction & PFN
        t0 = time.perf_counter()
        pts_xyz = voxels[:, :, :3]
        pts_mean = pts_xyz.sum(axis=1, keepdims=True) / np.maximum(voxel_num_points[:, None, None], 1).astype(np.float32)
        f_cluster = pts_xyz - pts_mean

        f_center = np.zeros_like(pts_xyz)
        x_offset = self.VOXEL_SIZE[0] / 2.0 + min_b[0]
        y_offset = self.VOXEL_SIZE[1] / 2.0 + min_b[1]
        z_offset = self.VOXEL_SIZE[2] / 2.0 + min_b[2]

        f_center[:, :, 0] = pts_xyz[:, :, 0] - (coords[:, 3:4] * self.VOXEL_SIZE[0] + x_offset)
        f_center[:, :, 1] = pts_xyz[:, :, 1] - (coords[:, 2:3] * self.VOXEL_SIZE[1] + y_offset)
        f_center[:, :, 2] = pts_xyz[:, :, 2] - (coords[:, 1:2] * self.VOXEL_SIZE[2] + z_offset)

        # Concatenate 11 channels
        features = np.concatenate([voxels, f_cluster, f_center], axis=-1)  # (M, 20, 11)

        # Zero out padding points using indicator mask
        mask_padding = (np.arange(self.MAX_POINTS_PER_VOXEL)[None, :] < voxel_num_points[:, None]).astype(np.float32)
        features *= mask_padding[:, :, None]

        # 5. PFN Processing (Linear -> BatchNorm1d -> ReLU -> MaxPool)
        linear_out = np.matmul(features, self._pfn_weight.T)  # (M, 20, 64)
        norm_out = (linear_out - self._pfn_norm_mean) / np.sqrt(self._pfn_norm_var + self._pfn_norm_eps) * self._pfn_norm_weight + self._pfn_norm_bias
        relu_out = np.maximum(norm_out, 0.0)
        pillar_features = np.max(relu_out, axis=1)  # (M, 64)

        # 6. BEV Pseudo-Image Scatter
        spatial_features = np.zeros((1, 64, self.GRID_SIZE[1], self.GRID_SIZE[0]), dtype=np.float32)
        spatial_features[0, :, coords[:, 2], coords[:, 3]] = pillar_features
        t_pfn_ms = (time.perf_counter() - t0) * 1000.0

        # 7. ONNX Runtime Backbone & Head Inference
        t0 = time.perf_counter()
        ort_outputs = self._session.run(None, {self._input_name: spatial_features})
        t_onnx_ms = (time.perf_counter() - t0) * 1000.0

        # 8. Extract Outputs & Decode Boxes
        t0 = time.perf_counter()
        cls_preds_raw = [ort_outputs[2 * i][0] for i in range(6)]   # list of (N_i, num_classes_i)
        box_preds_raw = [ort_outputs[2 * i + 1][0] for i in range(6)]  # list of (N_i, 10)
        all_box_encodings = np.concatenate(box_preds_raw, axis=0)  # (327680, 10)

        # Residual box decoding
        decoded_boxes = self._decode_boxes(all_box_encodings, self._anchors)

        # Sigmoid on classification logits
        cls_sigmoid = [1.0 / (1.0 + np.exp(-cp)) for cp in cls_preds_raw]

        # Count candidate boxes over threshold
        total_candidates = sum(int(np.sum(cs >= self.score_thresh)) for cs in cls_sigmoid)
        stats["raw_candidates"] = total_candidates
        t_decode_ms = (time.perf_counter() - t0) * 1000.0

        # 9. Multi-Class Rotated BEV NMS
        t0 = time.perf_counter()
        final_detections = self._multi_class_rotated_nms(cls_sigmoid, decoded_boxes)
        stats["final_detections"] = len(final_detections)
        t_nms_ms = (time.perf_counter() - t0) * 1000.0

        t_total_ms = (time.perf_counter() - t_start) * 1000.0
        stats["timings_ms"] = {
            "point_prep_ms": round(t_prep_ms, 2),
            "voxelisation_ms": round(t_voxel_ms, 2),
            "pfn_scatter_ms": round(t_pfn_ms, 2),
            "onnx_inference_ms": round(t_onnx_ms, 2),
            "decoding_ms": round(t_decode_ms, 2),
            "nms_ms": round(t_nms_ms, 2),
            "total_time_ms": round(t_total_ms, 2),
        }

        return PointPillarsResult(
            detections=final_detections,
            stats=stats,
            status="SUCCESS",
            error_message=None,
        )

    def _decode_boxes(self, box_encodings: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        """
        Applies exact OpenPCDet ResidualCoder decoding to residual box predictions.
        box_encodings: (N, 10) [xt, yt, zt, dxt, dyt, dzt, cost, sint, vxt, vyt]
        anchors: (N, 10) [xa, ya, za, dxa, dya, dza, ra, vxa=0, vya=0, ...]
        """
        xa, ya, za = anchors[:, 0], anchors[:, 1], anchors[:, 2]
        dxa, dya, dza = anchors[:, 3], anchors[:, 4], anchors[:, 5]
        ra = anchors[:, 6]

        xt, yt, zt = box_encodings[:, 0], box_encodings[:, 1], box_encodings[:, 2]
        dxt, dyt, dzt = box_encodings[:, 3], box_encodings[:, 4], box_encodings[:, 5]
        cost, sint = box_encodings[:, 6], box_encodings[:, 7]
        vxt, vyt = box_encodings[:, 8], box_encodings[:, 9]

        diagonal = np.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = np.exp(dxt) * dxa
        dyg = np.exp(dyt) * dya
        dzg = np.exp(dzt) * dza

        rg_cos = cost + np.cos(ra)
        rg_sin = sint + np.sin(ra)
        rg = np.arctan2(rg_sin, rg_cos)

        vxg = vxt
        vyg = vyt

        return np.stack([xg, yg, zg, dxg, dyg, dzg, rg, vxg, vyg], axis=-1)

    def _multi_class_rotated_nms(
        self,
        cls_scores_list: List[np.ndarray],
        decoded_boxes: np.ndarray,
    ) -> List[PointPillarsDetection]:
        """Performs class-independent greedy rotated BEV NMS across all 6 heads."""
        all_detections: List[PointPillarsDetection] = []
        start_idx = 0

        for head_idx, (cls_scores, global_label_map) in enumerate(zip(cls_scores_list, HEAD_CLASS_MAPPINGS)):
            num_anchors = cls_scores.shape[0]
            head_boxes = decoded_boxes[start_idx: start_idx + num_anchors]
            start_idx += num_anchors

            num_classes = cls_scores.shape[1]
            for k in range(num_classes):
                scores_k = cls_scores[:, k]
                mask = scores_k >= self.score_thresh
                if not np.any(mask):
                    continue

                cand_scores = scores_k[mask]
                cand_boxes = head_boxes[mask]

                # Sort topk
                order = np.argsort(-cand_scores)[:self.nms_pre_maxsize]

                # Greedy rotated NMS
                keep: List[int] = []
                while len(order) > 0 and len(keep) < self.nms_post_maxsize:
                    top_idx = order[0]
                    keep.append(top_idx)
                    if len(order) == 1:
                        break
                    ious = np.array([rotated_bev_iou(cand_boxes[top_idx], cand_boxes[o]) for o in order[1:]])
                    valid = ious <= self.nms_thresh
                    order = order[1:][valid]

                global_class_id = global_label_map[k]
                class_name = POINTPILLARS_CLASS_NAMES[global_class_id - 1]

                for sel_idx in keep:
                    b = cand_boxes[sel_idx]
                    s = float(cand_scores[sel_idx])
                    all_detections.append(PointPillarsDetection(
                        x=float(b[0]),
                        y=float(b[1]),
                        z=float(b[2]),
                        length=float(b[3]),
                        width=float(b[4]),
                        height=float(b[5]),
                        yaw=float(b[6]),
                        velocity_x=float(b[7]),
                        velocity_y=float(b[8]),
                        class_name=class_name,
                        class_id=global_class_id,
                        score=s,
                    ))

        return all_detections
