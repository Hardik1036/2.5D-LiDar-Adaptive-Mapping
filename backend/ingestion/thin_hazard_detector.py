"""
Millimeter-Residual Thin Hazard & Spike Strip Detector (Feature F1.6).
Detects low-profile dangerous obstacles (spike strips, metal plates, cables, rails)
sitting 1.0 cm to 8.0 cm above the local fitted ground plane with high reflection contrast,
powered by the ThreatNet1D 1D-ResNet ONNX model trained on RELLIS-3D.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("ThinHazardDetector")

# Optional ONNX Runtime import with graceful fallback
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    ort = None
    HAS_ORT = False


class ThinHazardDetector:
    """
    Evaluates orthogonal millimeter residuals against local ground planes
    to expose hazards too low to trigger standard step-height obstacle segmentation
    (10 mm - 80 mm elevation delta).
    
    Integrates ThreatNet1D ONNX inference with dynamic threshold calibration (0.75)
    and a vectorized NumPy fallback.
    """

    DEFAULT_ONNX_PATHS = [
        Path(__file__).resolve().parent.parent / "models" / "ThreatNet1D.onnx",
        Path("backend/models/ThreatNet1D.onnx"),
    ]

    DEFAULT_CALIB_PATHS = [
        Path(__file__).resolve().parent.parent / "models" / "threshold_calibration.json",
        Path("backend/models/threshold_calibration.json"),
    ]

    def __init__(
        self,
        min_residual: float = 0.010,   # 10 mm minimum height above ground
        max_residual: float = 0.080,   # 80 mm maximum height above ground
        min_points_cluster: int = 5,
        cluster_eps: float = 0.35,
        forced_cost: int = 255,
        model_path: Optional[str] = None,
        calibration_path: Optional[str] = None,
        use_onnx: bool = True,
    ):
        self.min_res = min_residual
        self.max_res = max_residual
        self.min_pts = min_points_cluster
        self.cluster_eps = cluster_eps
        self.forced_cost = forced_cost
        self.use_onnx = use_onnx

        # 1. Load optimal threshold from calibration JSON (default 0.75)
        self.threshold = self._load_threshold(calibration_path)

        # Pre-allocated inference buffer for zero-allocation runs
        self._inp_buf = np.zeros((1, 2, 512), dtype=np.float32)

        # 2. Initialize ONNX runtime session
        self.session: Optional[Any] = None
        self.input_name: str = "input"
        self.output_name: str = "threat_logits"
        if self.use_onnx and HAS_ORT:
            self._init_onnx_session(model_path)

    def _load_threshold(self, calib_path: Optional[str] = None) -> float:
        """Dynamically loads optimal threshold from calibration JSON."""
        search_paths = [Path(calib_path)] if calib_path else self.DEFAULT_CALIB_PATHS
        for p in search_paths:
            if p and p.exists() and p.is_file():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        thresh = float(data.get("threshold", 0.75))
                        logger.info(f"Loaded ThreatNet1D threshold={thresh:.4f} from {p}")
                        return thresh
                except Exception as e:
                    logger.warning(f"Error reading calibration file {p}: {e}")
        return 0.75

    def _init_onnx_session(self, model_path: Optional[str] = None) -> None:
        """Loads ThreatNet1D ONNX model into an InferenceSession."""
        search_paths = [Path(model_path)] if model_path else self.DEFAULT_ONNX_PATHS
        resolved_path = None
        for p in search_paths:
            if p and p.exists() and p.is_file():
                resolved_path = str(p)
                break

        if not resolved_path:
            logger.warning("ThreatNet1D.onnx model file not found. Falling back to NumPy inference.")
            return

        try:
            available_providers = ort.get_available_providers()
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available_providers else ["CPUExecutionProvider"]
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 1
            sess_opts.inter_op_num_threads = 1
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(resolved_path, sess_opts, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            logger.info(f"ThreatNet1D ONNX session initialized successfully from {resolved_path}")
        except Exception as e:
            logger.warning(f"Failed to load ONNX session from {resolved_path}: {e}. Falling back to NumPy.")
            self.session = None

    def fit_ground_plane(self, ground_points: np.ndarray) -> np.ndarray:
        """
        Fast covariance eigen-decomposition plane fit returning [a, b, c, d] normalized plane coefficients.
        Subsamples to <= 300 points for sub-0.05ms execution.
        """
        if len(ground_points) < 3:
            return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        xyz = ground_points[:, :3]
        step_s = max(1, len(xyz) // 300)
        sample = xyz[::step_s]
        centroid = np.mean(sample, axis=0)
        centered = sample - centroid

        cov = np.dot(centered.T, centered) / len(centered)
        _, vecs = np.linalg.eigh(cov)
        normal = vecs[:, 0]  # Smallest eigenvalue eigenvector

        # Orient normal upwards (+Z)
        if normal[2] < 0:
            normal = -normal

        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-6:
            normal = normal / norm_len

        d = -float(np.dot(normal, centroid))
        return np.array([normal[0], normal[1], normal[2], d], dtype=np.float32)

    def predict_probabilities(
        self,
        ground_points: np.ndarray,
        plane_model: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes ThreatNet1D threat probabilities for candidate near-ground points.

        Args:
            ground_points: (N, 3) or (N, 4) LiDAR points.
            plane_model: Optional [a, b, c, d] plane coefficients.

        Returns:
            Tuple of (cand_indices: np.ndarray, threat_probs: np.ndarray).
        """
        if ground_points is None or len(ground_points) < self.min_pts:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        xyz = ground_points[:, :3]

        if plane_model is None or len(plane_model) < 4:
            plane_model = self.fit_ground_plane(ground_points)

        a, b, c, d = plane_model
        denom = np.sqrt(a * a + b * b + c * c)
        if denom < 1e-6:
            denom = 1.0

        # Vectorized orthogonal signed distance
        signed_dist = (a * xyz[:, 0] + b * xyz[:, 1] + c * xyz[:, 2] + d) / denom

        # Candidate selection: near-ground points (min_residual <= delta_z <= max_residual)
        cand_mask = (signed_dist >= self.min_res) & (signed_dist <= self.max_res)
        cand_indices = np.nonzero(cand_mask)[0]

        if len(cand_indices) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        cand_dz = signed_dist[cand_indices]
        
        # Intensity channel extraction and normalization in [0, 1]
        if ground_points.shape[1] >= 4:
            cand_int = ground_points[cand_indices, 3]
            max_val = np.max(ground_points[:, 3]) if len(ground_points) > 0 else 1.0
            if max_val > 1.0:
                cand_int = np.clip(cand_int / 255.0, 0.0, 1.0).astype(np.float32)
            else:
                cand_int = np.clip(cand_int, 0.0, 1.0).astype(np.float32)
        else:
            cand_int = np.full(len(cand_indices), 0.2, dtype=np.float32)

        # Scale features for ThreatNet1D receptive field
        norm_dz = cand_dz * 150.0
        norm_i = cand_int * 16.0

        # 3. ONNX Runtime batch inference with ground baseline slot buffers
        if self.session is not None:
            K = len(cand_indices)
            slot_size = 5
            pad = 15
            step = slot_size + pad  # 20
            slots_per_buf = (512 - 20) // step  # 24 slots = 120 points per buffer

            threat_probs = np.zeros(K, dtype=np.float32)

            # Cap evaluation to top 120 candidates for a single 0.28ms buffer
            eval_limit = min(K, 120)
            if K > eval_limit:
                top_order = np.argpartition(cand_int, -eval_limit)[-eval_limit:]
                eval_idx = top_order
            else:
                eval_idx = np.arange(K)

            sub_dz = norm_dz[eval_idx]
            sub_i = norm_i[eval_idx]
            n_slots = len(eval_idx) // slot_size
            if n_slots == 0:
                return cand_indices, threat_probs

            self._inp_buf.fill(0)
            for s in range(n_slots):
                buf_pos = 10 + s * step
                self._inp_buf[0, 0, buf_pos:buf_pos+slot_size] = sub_dz[s*slot_size:(s+1)*slot_size]
                self._inp_buf[0, 1, buf_pos:buf_pos+slot_size] = sub_i[s*slot_size:(s+1)*slot_size]

            logits = self.session.run(None, {self.input_name: self._inp_buf})[0]
            probs = 1.0 / (1.0 + np.exp(-logits[0]))

            for s in range(n_slots):
                buf_pos = 10 + s * step
                slot_max = float(np.max(probs[buf_pos:buf_pos+slot_size]))
                orig_slice = eval_idx[s*slot_size:(s+1)*slot_size]
                threat_probs[orig_slice] = slot_max

            return cand_indices, threat_probs

        # 4. Fast Vectorized NumPy Fallback
        z_contrast = (norm_dz / 15.0) * 0.4 + (norm_i / 4.0) * 0.6
        fallback_probs = 1.0 / (1.0 + np.exp(-(z_contrast - 2.5)))
        return cand_indices, fallback_probs.astype(np.float32)

    def detect_threats(
        self,
        ground_points: np.ndarray,
        plane_model: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Identifies ground-level thin hazards using ThreatNet1D ONNX inference.

        Args:
            ground_points: (N, 3) or (N, 4) LiDAR ground points.
            plane_model: Optional [a, b, c, d] plane coefficients.

        Returns:
            List of detected thin hazard dictionaries with bounding boxes and lethal costs.
        """
        if ground_points is None or len(ground_points) < self.min_pts:
            return []

        cand_indices, threat_probs = self.predict_probabilities(ground_points, plane_model)
        if len(cand_indices) == 0:
            return []

        # Filter by calibrated optimal threshold (0.75)
        threat_mask = threat_probs >= self.threshold
        confirmed_indices = cand_indices[threat_mask]

        if len(confirmed_indices) < self.min_pts:
            return []

        cand_pts = ground_points[confirmed_indices, :3]
        conf_probs = threat_probs[threat_mask]

        # Fast spatial binning into clusters in < 0.2 ms
        inv_eps = 1.0 / self.cluster_eps
        clusters_map: Dict[Tuple[int, int], List[Tuple[np.ndarray, float]]] = {}
        for idx in range(len(cand_pts)):
            p = cand_pts[idx]
            pr = float(conf_probs[idx])
            cell_key = (int(np.floor(p[0] * inv_eps)), int(np.floor(p[1] * inv_eps)))
            clusters_map.setdefault(cell_key, []).append((p, pr))

        hazards: List[Dict[str, Any]] = []
        for cell_key, items in clusters_map.items():
            if len(items) >= self.min_pts:
                pts_arr = np.array([item[0] for item in items])
                probs_arr = [item[1] for item in items]
                min_x, max_x = float(np.min(pts_arr[:, 0])), float(np.max(pts_arr[:, 0]))
                min_y, max_y = float(np.min(pts_arr[:, 1])), float(np.max(pts_arr[:, 1]))
                min_z, max_z = float(np.min(pts_arr[:, 2])), float(np.max(pts_arr[:, 2]))
                cx = float(np.mean(pts_arr[:, 0]))
                cy = float(np.mean(pts_arr[:, 1]))
                cz = float(np.mean(pts_arr[:, 2]))

                hazards.append({
                    "type": "thin_hazard",
                    "centroid": (cx, cy, cz),
                    "bbox": (min_x, max_x, min_y, max_y, min_z, max_z),
                    "point_count": len(pts_arr),
                    "cost": self.forced_cost,
                    "radius": float(max(max_x - min_x, max_y - min_y) * 0.5 + 0.1),
                    "threat_probability": float(max(probs_arr)),
                })
                if len(hazards) >= 20:
                    break

        return hazards

    def detect_low_profile_hazards(
        self,
        ground_points: np.ndarray,
        plane_model: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Backward-compatible alias for detect_threats.
        """
        return self.detect_threats(ground_points, plane_model)

    def apply_hazards_to_leaves(
        self,
        leaves: List[Any],
        hazards: List[Dict[str, Any]],
    ) -> None:
        """
        Marks Quadtree leaves overlapping thin hazards as confirmed lethal (cost = 255).
        """
        if not leaves or not hazards:
            return

        for hz in hazards:
            cx, cy, _ = hz["centroid"]
            r2 = hz["radius"] * hz["radius"]
            hz_cost = hz.get("cost", self.forced_cost)

            for leaf in leaves:
                dist2 = (leaf.x - cx) ** 2 + (leaf.y - cy) ** 2
                if dist2 <= r2:
                    leaf.cost = max(leaf.cost, hz_cost)
