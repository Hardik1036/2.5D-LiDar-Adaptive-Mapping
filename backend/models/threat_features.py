"""
Ground residual + intensity feature computation for ThreatNet1D.
residual = point's Z minus the estimated local ground height at that (X,Y).
Ground height is estimated via a coarse grid (per-cell minimum Z), matching
the "ground residual" concept from the playbook (Section 7 / ThreatNet1D).
"""
import numpy as np

GRID_CELL_SIZE = 1.0  # meters
RESIDUAL_CLIP_MIN = -0.3
RESIDUAL_CLIP_MAX = 1.2


def compute_ground_residual(points, cell_size=GRID_CELL_SIZE):
    """
    points: (N, 4) [x, y, z, intensity]
    Returns: residual (N,) float32, norm_intensity (N,) float32 in [0,1]
    """
    x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]

    cell_x = np.floor(x / cell_size).astype(np.int64)
    cell_y = np.floor(y / cell_size).astype(np.int64)
    cell_key = cell_x * 100003 + cell_y  # unique-ish hash per cell

    # Per-cell minimum Z = local ground estimate
    order = np.argsort(cell_key)
    sorted_keys = cell_key[order]
    sorted_z = z[order]

    unique_keys, start_idx = np.unique(sorted_keys, return_index=True)
    ground_per_cell = np.minimum.reduceat(sorted_z, start_idx)

    ground_lookup = dict(zip(unique_keys.tolist(), ground_per_cell.tolist()))
    ground_z = np.array([ground_lookup[k] for k in cell_key], dtype=np.float32)

    residual = z - ground_z
    residual = np.clip(residual, RESIDUAL_CLIP_MIN, RESIDUAL_CLIP_MAX)

    i_min, i_max = intensity.min(), intensity.max()
    if i_max - i_min < 1e-6:
        norm_intensity = np.zeros_like(intensity, dtype=np.float32)
    else:
        norm_intensity = ((intensity - i_min) / (i_max - i_min)).astype(np.float32)

    return residual.astype(np.float32), norm_intensity


def make_threat_windows(residual, norm_intensity, window_size=512, stride=512):
    """
    ThreatNet1D expects input shape (1, 2, 512) — chunks the point stream into
    fixed-size windows of [residual, intensity] pairs.
    Returns: windows (num_windows, 2, window_size), and the point-index range each window covers.
    """
    n = len(residual)
    windows = []
    ranges = []
    for start in range(0, n, stride):
        end = start + window_size
        r = residual[start:end]
        i = norm_intensity[start:end]
        if len(r) < window_size:
            pad = window_size - len(r)
            r = np.pad(r, (0, pad), mode='constant', constant_values=0)
            i = np.pad(i, (0, pad), mode='constant', constant_values=0)
        windows.append(np.stack([r, i], axis=0))
        ranges.append((start, min(end, n)))
    return np.stack(windows, axis=0).astype(np.float32), ranges
