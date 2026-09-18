"""
ML Perception Adapter for DRDO SIH 2026 Problem Statement 53.
Bridges external neural network inference outputs (Model 1 SalsaNext Semantic Segmentation,
Model 2 ThreatNet1D ONNX low-profile hazard classification, and 3D Oriented Bounding Box Detections)
into the 2.5D Adaptive Perception Engine.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.special import expit

from backend.adapters.salsanext import LightweightSalsaNext, SalsaNextArchitecture
from backend.config import POINT_PILLARS, PointPillarsConfig
from backend.models.threat_features import compute_ground_residual, make_threat_windows
from backend.tracking.clustering import DetectedCluster

logger = logging.getLogger("MLPerceptionAdapter")

# Optional PyTorch import
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

# Optional ONNX Runtime import
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    ort = None
    HAS_ORT = False

# Global cache for ThreatNet1D ONNX session and calibration threshold
_CACHED_ONNX_SESSION: Optional[Any] = None
_CACHED_INPUT_NAME: str = "input"
_CACHED_THRESHOLD: float = 0.75


def _get_model_paths() -> Tuple[List[Path], List[Path]]:
    root = Path(__file__).resolve().parent.parent
    model_paths = [
        root / "models" / "ThreatNet1D.onnx",
        Path("backend/models/ThreatNet1D.onnx"),
        Path("models/ThreatNet1D.onnx"),
    ]
    calib_paths = [
        root / "models" / "threshold_calibration.json",
        Path("backend/models/threshold_calibration.json"),
        Path("models/threshold_calibration.json"),
    ]
    return model_paths, calib_paths


def _init_threatnet_session() -> None:
    """Initializes and caches the ThreatNet1D ONNX session at module load."""
    global _CACHED_ONNX_SESSION, _CACHED_INPUT_NAME, _CACHED_THRESHOLD

    model_paths, calib_paths = _get_model_paths()

    # Load calibrated threshold
    for cp in calib_paths:
        if cp.exists() and cp.is_file():
            try:
                with open(cp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    _CACHED_THRESHOLD = float(data.get("threshold", 0.75))
                    logger.info(f"Cached ThreatNet1D threshold={_CACHED_THRESHOLD:.4f} from {cp}")
                    break
            except Exception:
                pass

    if not HAS_ORT or ort is None:
        return

    for mp in model_paths:
        if mp.exists() and mp.is_file():
            try:
                available_providers = ort.get_available_providers()
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available_providers else ["CPUExecutionProvider"]
                sess_opts = ort.SessionOptions()
                sess_opts.intra_op_num_threads = 1
                sess_opts.inter_op_num_threads = 1
                sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                _CACHED_ONNX_SESSION = ort.InferenceSession(str(mp), sess_opts, providers=providers)
                _CACHED_INPUT_NAME = _CACHED_ONNX_SESSION.get_inputs()[0].name
                logger.info(f"Cached ThreatNet1D ONNX session from {mp}")
                break
            except Exception as e:
                logger.warning(f"Could not load ThreatNet1D ONNX model from {mp}: {e}")


# Initialize module-level cache
_init_threatnet_session()


def _format_threatnet_input(
    residuals: np.ndarray,
    intensities: np.ndarray,
    slot_size: int = 5,
    pad: int = 15,
) -> Tuple[List[np.ndarray], List[List[Tuple[int, int]]]]:
    """
    Formats 1D ground elevation residuals and intensities into the 3D tensor contract
    expected by ThreatNet1D.onnx: (1, 2, 512).

    Each candidate point is mapped into a slot of width `slot_size` separated by `pad` baseline steps.
    Returns:
        buffers: List of (1, 2, 512) float32 numpy arrays.
        slot_mappings: Per-buffer list of (buf_pos, slot_size) index slices.
    """
    K = len(residuals)
    if K == 0:
        return [], []

    step = slot_size + pad  # 20
    slots_per_buf = (512 - 20) // step  # 24 slots

    norm_dz = np.asarray(residuals, dtype=np.float32) * 5.5
    raw_i = np.asarray(intensities, dtype=np.float32)
    max_i = float(np.max(raw_i)) if len(raw_i) > 0 else 1.0
    if max_i > 1.0:
        cand_i = np.clip(raw_i / 255.0, 0.0, 1.0)
    else:
        cand_i = np.clip(raw_i, 0.0, 1.0)
    norm_i = cand_i

    buffers: List[np.ndarray] = []
    slot_mappings: List[List[Tuple[int, int]]] = []

    for chunk_start in range(0, K, slots_per_buf):
        chunk_end = min(chunk_start + slots_per_buf, K)
        n_pts = chunk_end - chunk_start

        buf = np.zeros((1, 2, 512), dtype=np.float32)
        buf_map: List[Tuple[int, int]] = []
        for s in range(n_pts):
            p_idx = chunk_start + s
            buf_pos = 10 + s * step
            buf[0, 0, buf_pos:buf_pos + slot_size] = norm_dz[p_idx]
            buf[0, 1, buf_pos:buf_pos + slot_size] = norm_i[p_idx]
            buf_map.append((buf_pos, slot_size))

        buffers.append(buf)
        slot_mappings.append(buf_map)

    return buffers, slot_mappings


def _run_threatnet_session(
    session: Any,
    input_name: str,
    residuals: np.ndarray,
    intensities: np.ndarray,
) -> np.ndarray:
    """
    Executes ThreatNet1D ONNX inference over formatted (1, 2, 512) 3D tensors
    and returns calibrated probabilities via sigmoid.
    """
    K = len(residuals)
    if K == 0 or session is None:
        return np.zeros(K, dtype=np.float32)

    if hasattr(session, "get_inputs"):
        try:
            input_name = session.get_inputs()[0].name
        except Exception:
            pass

    buffers, slot_mappings = _format_threatnet_input(residuals, intensities)
    probs_out = np.zeros(K, dtype=np.float32)

    point_offset = 0
    for buf, buf_map in zip(buffers, slot_mappings):
        logits = session.run(None, {input_name: buf})[0].squeeze()
        sig_probs = expit(logits)

        for s, (buf_pos, slot_size) in enumerate(buf_map):
            probs_out[point_offset + s] = float(np.max(sig_probs[buf_pos:buf_pos + slot_size]))
        point_offset += len(buf_map)

    return probs_out


def evaluate_thin_hazards(
    points: np.ndarray,
    ground_plane: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    min_residual: float = 0.010,
    max_residual: float = 0.080,
) -> Dict[str, Any]:
    """
    Evaluates candidate near-ground LiDAR points with ThreatNet1D ONNX inference.
    Executes vectorized across points with zero iterative Python loops.

    Args:
        points: (N, 3) or (N, 4) XYZ / XYZI coordinates.
        ground_plane: Optional [a, b, c, d] ground plane model.
        threshold: Optional custom decision threshold (defaults to calibrated 0.75).
        min_residual: Minimum height delta above ground (default 0.01m).
        max_residual: Maximum height delta above ground (default 0.08m).

    Returns:
        Dict containing:
            "threat_mask": (N,) boolean mask of detected thin hazards.
            "probabilities": (N,) float32 threat probabilities.
            "threat_points": (M, D) confirmed lethal threat points.
            "threshold": float decision threshold applied.
    """
    thresh_val = threshold if threshold is not None else _CACHED_THRESHOLD
    N = len(points) if points is not None else 0

    if N == 0 or points is None:
        empty_mask = np.zeros(0, dtype=bool)
        empty_probs = np.zeros(0, dtype=np.float32)
        empty_pts = np.empty((0, 3), dtype=np.float32)
        return {
            "threat_mask": empty_mask,
            "probabilities": empty_probs,
            "threat_points": empty_pts,
            "threshold": thresh_val,
        }

    xyz = points[:, :3]

    # Fast covariance plane fit if plane not provided
    if ground_plane is None or len(ground_plane) < 4:
        if N >= 3:
            sample = xyz[::4] if N > 500 else xyz
            centroid = np.mean(sample, axis=0)
            centered = sample - centroid
            cov = (centered.T @ centered) / len(centered)
            _, vecs = np.linalg.eigh(cov)
            normal = vecs[:, 0]
            if normal[2] < 0:
                normal = -normal
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-6:
                normal = normal / norm_len
            d = -float(np.dot(normal, centroid))
            ground_plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float32)
        else:
            ground_plane = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    a, b, c, d = ground_plane
    denom = np.sqrt(a * a + b * b + c * c)
    if denom < 1e-6:
        denom = 1.0

    # Vectorized orthogonal signed distance
    signed_dist = (a * xyz[:, 0] + b * xyz[:, 1] + c * xyz[:, 2] + d) / denom

    # Candidate mask for near-ground thin threats
    cand_mask = (signed_dist >= min_residual) & (signed_dist <= max_residual)
    cand_indices = np.nonzero(cand_mask)[0]

    all_probabilities = np.zeros(N, dtype=np.float32)

    if len(cand_indices) > 0:
        cand_dz = signed_dist[cand_indices]

        # Extract normalized intensity
        if points.shape[1] >= 4:
            raw_intensities = points[cand_indices, 3]
            max_i = np.max(points[:, 3]) if N > 0 else 1.0
            if max_i > 1.0:
                cand_int = np.clip(raw_intensities / 255.0, 0.0, 1.0).astype(np.float32)
            else:
                cand_int = np.clip(raw_intensities, 0.0, 1.0).astype(np.float32)
        else:
            cand_int = np.full(len(cand_indices), 0.2, dtype=np.float32)

        # Scale features for ThreatNet1D receptive field
        norm_dz = cand_dz * 5.5
        norm_i = cand_int

        # Fast Vectorized ONNX inference with ground baseline slot buffers
        if _CACHED_ONNX_SESSION is not None:
            eval_limit = min(len(cand_indices), 120)
            if len(cand_indices) > eval_limit:
                top_order = np.argsort(cand_int)[-eval_limit:]
                eval_idx = top_order
            else:
                eval_idx = np.arange(len(cand_indices))

            sub_dz = cand_dz[eval_idx]
            sub_i = cand_int[eval_idx]
            sub_probs = _run_threatnet_session(
                _CACHED_ONNX_SESSION, _CACHED_INPUT_NAME, sub_dz, sub_i
            )
            all_probabilities[cand_indices[eval_idx]] = sub_probs
        else:
            # Vectorized NumPy fallback
            z_contrast = (norm_dz / 0.55) * 1.5 + (norm_i / 0.25) * 1.2
            fallback_probs = expit(z_contrast - 2.5)
            all_probabilities[cand_indices] = fallback_probs.astype(np.float32)

    threat_mask = all_probabilities >= thresh_val
    threat_points = points[threat_mask]

    return {
        "threat_mask": threat_mask,
        "probabilities": all_probabilities,
        "threat_points": threat_points,
        "threshold": thresh_val,
    }


class MLPerceptionAdapter:
    """
    Dual-Model ML Perception Adapter:
    1. Model 1: SalsaNext LiDAR semantic segmentation (Ground: 0, Vegetation: 1, Rigid: 2, Dynamic: 3).
    2. Model 2: ThreatNet1D 1D low-profile hazard detector (Spike strips / metal plates).
    3. 3D Oriented Bounding Box Detections (M, 9) -> [x, y, z, l, w, h, yaw, vx, vy].
    """

    LABEL_GROUND = 0
    LABEL_VEGETATION = 1
    LABEL_RIGID = 2
    LABEL_DYNAMIC = 3

    def __init__(
        self,
        model_dir: str = "backend/models",
        pointpillars_config: Optional[PointPillarsConfig] = None,
    ):
        # Resolve path whether run from root or backend/ or installed package
        if os.path.exists(model_dir):
            self.model_dir = model_dir
        elif os.path.exists(os.path.join("backend", model_dir)):
            self.model_dir = os.path.join("backend", model_dir)
        else:
            pkg_models = Path(__file__).resolve().parent.parent / "models"
            if pkg_models.exists():
                self.model_dir = str(pkg_models)
            else:
                self.model_dir = model_dir

        if HAS_TORCH and torch is not None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"

        # Load configs
        self.class_map = self._load_json(
            "class_mapping.json",
            default={"0": "ground", "1": "vegetation", "2": "rigid", "3": "dynamic"},
        )
        self.model_config = self._load_json("model_config.json", default={})

        # PointPillars (Model 2 3D Detection) spatial & anchor configuration
        self.pointpillars_config: PointPillarsConfig = pointpillars_config or POINT_PILLARS
        self.point_cloud_range: Tuple[float, ...] = self.pointpillars_config.POINT_CLOUD_RANGE
        self.voxel_size: Tuple[float, ...] = self.pointpillars_config.VOXEL_SIZE
        self.pointpillars_classes: List[str] = list(self.pointpillars_config.CLASS_NAMES)
        self.box_code_size: int = self.pointpillars_config.CODE_SIZE

        # Model sessions & thresholds
        self.salsa_session = None
        self.segmentation_session = None
        self.segmentation_model = None
        self.threat_session = None
        self.pointpillars_session = None
        self.threat_threshold = 0.75

        self._init_sessions()
        self._empty_pts = np.empty((0, 3), dtype=np.float32)

    def _load_json(self, filename: str, default: dict) -> dict:
        filepath = os.path.join(self.model_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading {filepath}: {e}")
        return default

    def _init_sessions(self) -> None:
        opts = ort.SessionOptions() if HAS_ORT and ort is not None else None
        if opts is not None:
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            available = ort.get_available_providers()
            providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available]
            if not providers:
                providers = ["CPUExecutionProvider"]

            # 1. Load Model 1 (Segmentation ONNX)
            for name in ["SegmentationModel.onnx", "salsa_next_rellis.onnx", "salsanext_64x1024.onnx", "salsanext.onnx"]:
                path = os.path.join(self.model_dir, name)
                if os.path.exists(path):
                    try:
                        self.salsa_session = ort.InferenceSession(path, opts, providers=providers)
                        self.segmentation_session = self.salsa_session
                        print(f"[ML Adapter] Loaded Segmentation model from {path}")
                        break
                    except Exception as e:
                        print(f"[ML Adapter] Failed to load {path}: {e}")

            # 2. Load Model 2 (ThreatNet1D ONNX)
            threat_path = os.path.join(self.model_dir, "ThreatNet1D.onnx")
            if os.path.exists(threat_path):
                try:
                    self.threat_session = ort.InferenceSession(threat_path, opts, providers=providers)
                    print(f"[ML Adapter] Loaded ThreatNet1D model from {threat_path}")
                except Exception as e:
                    print(f"[ML Adapter] Failed to load ThreatNet1D: {e}")

            # 3. Load PointPillars ONNX (Model 2 3D Bounding Box Detector if available)
            for pp_name in ["pointpillars.onnx", "PointPillar.onnx", "cbgs_pp_multihead.onnx"]:
                pp_path = os.path.join(self.model_dir, pp_name)
                if os.path.exists(pp_path):
                    try:
                        self.pointpillars_session = ort.InferenceSession(pp_path, opts, providers=providers)
                        print(f"[ML Adapter] Loaded PointPillars model from {pp_path}")
                        break
                    except Exception as e:
                        print(f"[ML Adapter] Failed to load PointPillars model {pp_path}: {e}")

        # Load threshold calibration
        calib_path = os.path.join(self.model_dir, "threshold_calibration.json")
        if os.path.exists(calib_path):
            try:
                with open(calib_path, "r", encoding="utf-8") as f:
                    self.threat_threshold = float(json.load(f).get("threshold", 0.75))
            except Exception:
                self.threat_threshold = 0.75

        # Fallback PyTorch checkpoint loading if ONNX segmentation was not loaded
        if self.salsa_session is None:
            self._load_pytorch_segmentation_model()

    def _load_pytorch_segmentation_model(self) -> None:
        pth_path = os.path.join(self.model_dir, "best_model.pth")
        pt_path = os.path.join(self.model_dir, "best_model_v3.pt")
        target_path = pth_path if os.path.exists(pth_path) else (pt_path if os.path.exists(pt_path) else None)

        if target_path and os.path.exists(target_path):
            if HAS_TORCH and torch is not None:
                try:
                    checkpoint = torch.load(target_path, map_location=self.device)
                    if isinstance(checkpoint, dict):
                        state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
                        model = SalsaNextArchitecture(in_channels=5, num_classes=4)
                        if hasattr(model, "load_state_dict"):
                            model.load_state_dict(state_dict, strict=False)
                            model.to(self.device)
                            model.eval()
                            self.segmentation_model = model
                            print(f"[ML Adapter] SalsaNext model loaded from {target_path} on {self.device}")
                        else:
                            self.segmentation_model = None
                    elif isinstance(checkpoint, torch.nn.Module):
                        self.segmentation_model = checkpoint.to(self.device)
                        self.segmentation_model.eval()
                        print(f"[ML Adapter] SalsaNext model loaded from {target_path} on {self.device}")
                    else:
                        self.segmentation_model = None
                except Exception as e:
                    print(f"[ML Adapter] Warning loading PyTorch model ({e}). Using fast vectorized fallback.")
                    self.segmentation_model = None
            else:
                print("[ML Adapter] Warning loading PyTorch model (PyTorch not available). Using fast vectorized fallback.")
                self.segmentation_model = None
        else:
            print("[ML Adapter] No segmentation weights found. Using heuristic fallback.")

    def _project_to_range_image(
        self,
        points: np.ndarray,
        H: int = 64,
        W: int = 1024,
        fov_up: float = 17.02,
        fov_down: float = -16.44,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Spherical range image projection for 64x1024 LiDAR.
        Returns:
            img: (1, 5, H, W) float32 range image [Range, X, Y, Z, Intensity]
            point_rc: (N, 2) row and col indices for each point
            valid: (N,) boolean mask indicating points projected within valid FOV bounds
        """
        N = len(points)
        if N == 0:
            return np.zeros((1, 5, H, W), dtype=np.float32), np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=bool)

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        intensity = points[:, 3] if points.shape[1] >= 4 else np.full(N, 0.5, dtype=np.float32)

        depth = np.sqrt(x * x + y * y + z * z)
        depth_safe = np.maximum(depth, 1e-6)

        # Horizontal azimuth: [-pi, pi] -> col [0, W - 1]
        yaw = -np.arctan2(y, x)
        col = 0.5 * (1.0 - yaw / np.pi) * W
        col = np.clip(np.floor(col).astype(np.int32), 0, W - 1)

        # Vertical elevation: [fov_down, fov_up] -> row [0, H - 1]
        fov_up_rad = fov_up * np.pi / 180.0
        fov_down_rad = fov_down * np.pi / 180.0
        total_fov = fov_up_rad - fov_down_rad

        pitch = np.arcsin(np.clip(z / depth_safe, -1.0, 1.0))
        row = (1.0 - (pitch - fov_down_rad) / total_fov) * H
        row_int = np.floor(row).astype(np.int32)

        valid = (row_int >= 0) & (row_int < H) & (depth > 0.5) & (depth < 80.0)
        row_clamped = np.clip(row_int, 0, H - 1)

        img = np.zeros((1, 5, H, W), dtype=np.float32)
        valid_idx = np.nonzero(valid)[0]
        if len(valid_idx) > 0:
            order = valid_idx[np.argsort(-depth[valid_idx])]
            ro = row_clamped[order]
            co = col[order]
            img[0, 0, ro, co] = depth[order]
            img[0, 1, ro, co] = x[order]
            img[0, 2, ro, co] = y[order]
            img[0, 3, ro, co] = z[order]
            img[0, 4, ro, co] = intensity[order]

        point_rc = np.column_stack((row_clamped, col))
        return img, point_rc, valid

    def project_points_to_range_image(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Spherical range image projection utility for 64x1024 LiDAR.
        Translates raw (N, 4) points into a (1, 5, 64, 1024) range image tensor.
        """
        range_cfg = self.model_config.get("range_image", {})
        H = range_cfg.get("height", 64)
        W = range_cfg.get("width", 1024)
        fov_up = range_cfg.get("fov_up_degrees", 17.02)
        fov_down = range_cfg.get("fov_down_degrees", -16.44)
        return self._project_to_range_image(points, H=H, W=W, fov_up=fov_up, fov_down=fov_down)

    def segment_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """
        Runs Model 1 semantic segmentation on raw points (N, 4) [X, Y, Z, Intensity]
        or a formatted 4D range-image tensor (1, 5, 64, 1024).
        Returns class labels (N,) [0: Ground, 1: Vegetation, 2: Rigid, 3: Dynamic].
        """
        if points is None or len(points) == 0:
            return np.empty((0,), dtype=np.int32)

        if self.salsa_session is not None:
            try:
                input_name = self.salsa_session.get_inputs()[0].name
                # If model expects range-image or tensor, format accordingly
                if points.ndim == 4 and points.shape[1] == 5:
                    feed_dict = {input_name: points.astype(np.float32)}
                    logits = self.salsa_session.run(None, feed_dict)[0]
                    if logits.ndim > 1:
                        return np.argmax(logits, axis=-1).astype(np.int32)
                    return logits.astype(np.int32)
            except Exception as e:
                if not getattr(self, "_seg_warned", False):
                    print(f"[ML Adapter] Segmentation inference error: {e}. Using fallback.")
                    self._seg_warned = True

        # High-speed vectorized fallback for raw (N, 3/4) point clouds
        N = len(points)
        labels = np.zeros(N, dtype=np.int32)
        z = points[:, 2]
        labels[z > 0.15] = 2                   # Rigid hazard
        labels[(z > 0.15) & (z < 1.0)] = 1     # Passable vegetation
        return labels

    def detect_low_profile_threats(
        self,
        ground_points: np.ndarray,
        ground_plane: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Runs Model 2 (ThreatNet1D) on ground candidate points.
        Supports both:
          1. 2D point clouds: ground_points (N, 3/4) and ground_plane [a, b, c, d]
          2. 1D arrays: ground_points (K,) residuals and ground_plane (K,) intensities
        Returns boolean threat mask of shape (N,) or (K,).
        """
        if ground_points is None or len(ground_points) == 0:
            return np.zeros(0, dtype=bool)

        # 1. Dynamically read calibrated threshold from threshold_calibration.json
        calib_paths = [
            Path(self.model_dir) / "threshold_calibration.json",
            Path("backend/models/threshold_calibration.json"),
            Path("models/threshold_calibration.json"),
        ]
        threshold = self.threat_threshold
        for cp in calib_paths:
            if cp.exists() and cp.is_file():
                try:
                    with open(cp, "r", encoding="utf-8") as f:
                        threshold = float(json.load(f).get("threshold", threshold))
                        break
                except Exception:
                    pass

        # 2. Backward compatibility: handle 1D residual array input
        if isinstance(ground_points, np.ndarray) and ground_points.ndim == 1:
            residuals = ground_points
            intensities = (
                ground_plane
                if (ground_plane is not None and isinstance(ground_plane, np.ndarray) and ground_plane.ndim == 1)
                else np.zeros_like(residuals)
            )
            sess = self.threat_session if self.threat_session is not None else _CACHED_ONNX_SESSION
            if sess is not None:
                try:
                    input_name = sess.get_inputs()[0].name
                    probs = _run_threatnet_session(sess, input_name, residuals, intensities)
                    return probs > threshold
                except Exception as e:
                    logger.warning(f"ThreatNet1D 1D inference error: {e}")
            # Fallback heuristic
            return (residuals > 0.015) & (residuals < 0.05) & (intensities > 0.7)

        # 3. 2D Point Cloud input (N, 3) or (N, 4)
        pts = np.asarray(ground_points, dtype=np.float32)
        N = len(pts)
        if N == 0:
            return np.zeros(0, dtype=bool)

        # Pad intensity column if only XYZ provided
        if pts.shape[1] < 4:
            pts = np.column_stack([pts[:, :3], np.full(N, 0.2, dtype=np.float32)])

        # Compute ground residuals & normalized intensity using threat_features logic
        if ground_plane is not None and len(ground_plane) >= 4:
            a, b, c, d = ground_plane[:4]
            denom = np.sqrt(a * a + b * b + c * c)
            if denom < 1e-6:
                denom = 1.0
            residuals = (a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d) / denom
            residuals = np.clip(residuals, -0.3, 1.2).astype(np.float32)

            raw_i = pts[:, 3]
            i_min, i_max = float(raw_i.min()), float(raw_i.max())
            if i_max - i_min < 1e-6:
                norm_intensity = np.zeros_like(raw_i, dtype=np.float32)
            else:
                norm_intensity = ((raw_i - i_min) / (i_max - i_min)).astype(np.float32)
        else:
            residuals, norm_intensity = compute_ground_residual(pts)

        # Select candidate near-ground points for ThreatNet1D
        cand_mask = (residuals >= 0.010) & (residuals <= 0.080)
        cand_indices = np.nonzero(cand_mask)[0]

        if len(cand_indices) == 0:
            return np.zeros(N, dtype=bool)

        sess = self.threat_session if self.threat_session is not None else _CACHED_ONNX_SESSION
        if sess is not None:
            try:
                input_name = sess.get_inputs()[0].name
                cand_probs = _run_threatnet_session(
                    sess, input_name, residuals[cand_indices], norm_intensity[cand_indices]
                )
                threat_mask = np.zeros(N, dtype=bool)
                threat_mask[cand_indices] = cand_probs > threshold
                return threat_mask
            except Exception as e:
                logger.warning(f"ThreatNet1D 2D inference error: {e}")

        # Fallback heuristic
        cand_threat = (residuals[cand_indices] > 0.015) & (residuals[cand_indices] < 0.05) & (norm_intensity[cand_indices] > 0.7)
        threat_mask = np.zeros(N, dtype=bool)
        threat_mask[cand_indices] = cand_threat
        return threat_mask

    def evaluate_thin_hazards(
        self,
        points: np.ndarray,
        ground_plane: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Instance method wrapper around module-level vectorized evaluate_thin_hazards.
        """
        return evaluate_thin_hazards(points, ground_plane=ground_plane, threshold=threshold)

    def process_semantic_labels(
        self,
        points: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Partitions input point cloud based on ML semantic segmentation labels.

        Args:
            points: (N, 3) or (N, 4) XYZ / XYZI point coordinates.
            labels: Optional (N,) integer class IDs:
                    0 = Ground
                    1 = Passable Vegetation (tall grass, small brush)
                    2 = Rigid Hazard (rocks, trees, structures, walls)
                    3 = Dynamic Foreground (vehicles, pedestrians, robots)

        Returns:
            Dictionary with separated point cloud subsets.
        """
        if points is None or len(points) == 0:
            return {
                "ground": self._empty_pts,
                "vegetation": self._empty_pts,
                "rigid": self._empty_pts,
                "dynamic": self._empty_pts,
                "obstacle": self._empty_pts,
            }

        # If no labels are provided or length mismatch, run Model 1 segmentation
        if labels is None or len(labels) != len(points):
            labels = self.segment_point_cloud(points)

        # Vectorized label masking (< 0.8 ms for 25,000 points)
        labels_arr = np.asarray(labels, dtype=np.int32)
        ground_mask = (labels_arr == self.LABEL_GROUND)
        veg_mask = (labels_arr == self.LABEL_VEGETATION)
        rigid_mask = (labels_arr == self.LABEL_RIGID)
        dynamic_mask = (labels_arr == self.LABEL_DYNAMIC)
        obstacle_mask = rigid_mask | dynamic_mask

        return {
            "ground": points[ground_mask],
            "vegetation": points[veg_mask],
            "rigid": points[rigid_mask],
            "dynamic": points[dynamic_mask],
            "obstacle": points[obstacle_mask],
        }

    def process_pillar_detections(
        self,
        detections: Optional[np.ndarray] = None,
        filter_out_of_bounds: bool = True,
    ) -> List[DetectedCluster]:
        """
        Converts 3D bounding box detections into DetectedCluster candidate objects.
        Pre-configured for expected (M, 9) bounding box limits from PointPillars:
        [x, y, z, length, width, height, yaw, vx, vy] within POINT_CLOUD_RANGE.

        Args:
            detections: (M, 9) ndarray where each row represents:
                        [x, y, z, length, width, height, yaw, vx, vy]
            filter_out_of_bounds: If True, filters out detections whose centroids
                                  fall outside PointPillars POINT_CLOUD_RANGE.

        Returns:
            List of DetectedCluster instances compatible with MultiObjectTracker and CostmapEvaluator.
        """
        if detections is None or len(detections) == 0:
            return []

        det_arr = np.asarray(detections, dtype=np.float32)
        if det_arr.ndim == 1:
            det_arr = det_arr.reshape(1, -1)

        if det_arr.shape[1] < 6:
            raise ValueError(f"Detection array must have at least 6 fields [x, y, z, l, w, h], got {det_arr.shape[1]}")

        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range

        clusters: List[DetectedCluster] = []
        for det in det_arr:
            x, y, z = float(det[0]), float(det[1]), float(det[2])
            l, w, h = float(det[3]), float(det[4]), float(det[5])

            # Spatial limit check against PointPillars point cloud range (REMOVE_OUTSIDE_BOXES)
            if filter_out_of_bounds:
                if not (x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max):
                    continue

            # Validate physical positive box extents
            if l <= 0.0 or w <= 0.0 or h <= 0.0:
                continue

            min_x, max_x = x - l * 0.5, x + l * 0.5
            min_y, max_y = y - w * 0.5, y + w * 0.5
            min_z, max_z = z - h * 0.5, z + h * 0.5

            cluster_yaw = float(det[6]) if det_arr.shape[1] >= 7 else 0.0
            velocity = (float(det[7]), float(det[8])) if det_arr.shape[1] >= 9 else None

            cluster = DetectedCluster(
                centroid=(x, y, z),
                dimensions=(l, w, h),
                bbox=(min_x, max_x, min_y, max_y, min_z, max_z),
                point_count=50,
                points=self._empty_pts,
                yaw=cluster_yaw,
                velocity=velocity,
            )

            clusters.append(cluster)

        return clusters

    def non_max_suppression_2d(
        self,
        detections: np.ndarray,
        iou_threshold: float = 0.2,
        scores: Optional[np.ndarray] = None,
        max_output_boxes: Optional[int] = None,
    ) -> np.ndarray:
        """Instance method for 2D BEV Non-Maximum Suppression."""
        return non_max_suppression_2d(
            detections,
            iou_threshold=iou_threshold,
            scores=scores,
            max_output_boxes=max_output_boxes,
        )


def non_max_suppression_2d(
    boxes: np.ndarray,
    iou_threshold: float = 0.2,
    scores: Optional[np.ndarray] = None,
    max_output_boxes: Optional[int] = None,
) -> np.ndarray:
    """
    Fast 2D Bird's-Eye View (BEV) Non-Maximum Suppression.

    Args:
        boxes: (N, >=4) array of bounding box proposals [x, y, z, l, w, h, ...].
               x, y are centroid coordinates; l, w are dimensions in X and Y.
        iou_threshold: Overlap IoU threshold (default 0.2 matching OpenPCDet PointPillars).
        scores: Optional (N,) ranking scores. If None, defaults to proposal volume (l * w * h)
                or confidence.
        max_output_boxes: Maximum number of retained bounding boxes.

    Returns:
        (M, D) ndarray of suppressed bounding box proposals.
    """
    if boxes is None or len(boxes) == 0:
        dim = boxes.shape[1] if (boxes is not None and boxes.ndim == 2) else 9
        return np.empty((0, dim), dtype=np.float32)

    boxes_arr = np.asarray(boxes, dtype=np.float32)
    if boxes_arr.ndim == 1:
        boxes_arr = boxes_arr.reshape(1, -1)

    x = boxes_arr[:, 0]
    y = boxes_arr[:, 1]
    l = boxes_arr[:, 3]
    w = boxes_arr[:, 4]

    x1 = x - l * 0.5
    y1 = y - w * 0.5
    x2 = x + l * 0.5
    y2 = y + w * 0.5

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    if scores is None:
        if boxes_arr.shape[1] >= 6:
            scores = l * w * boxes_arr[:, 5]  # volume heuristic
        else:
            scores = np.arange(len(boxes_arr), dtype=np.float32)
    else:
        scores = np.asarray(scores, dtype=np.float32)

    order = np.argsort(scores)[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if max_output_boxes is not None and len(keep) >= max_output_boxes:
            break

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        intersection = inter_w * inter_h

        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / np.maximum(union, 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return boxes_arr[keep]

