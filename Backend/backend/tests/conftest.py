"""
Pytest fixtures providing in-memory procedural LiDAR sweeps for testing without disk-stored data.
Generates mathematically clean 5-channel [X, Y, Z, Intensity, Ring] sweeps.
"""

from pathlib import Path
import tempfile
import pytest
import numpy as np


def generate_procedural_sweep(frame_idx: int, dt: float = 0.04) -> np.ndarray:
    """
    Generates a procedural 5-channel [X, Y, Z, Intensity, Ring] sweep.
    - Planar asphalt ground with subtle texture.
    - Road curbs at y = +/- 4.0m with 0.10m step.
    - Moving dynamic vehicle moving forward along x at v = 3.5 m/s.
    - Static road obstacles (pole, tree).
    - Ring indices (0-63).
    """
    rng = np.random.RandomState(42 + frame_idx)

    # 1. Road surface
    n_road = 6000
    rx = rng.uniform(-15.0, 35.0, n_road).astype(np.float32)
    ry = rng.uniform(-4.0, 4.0, n_road).astype(np.float32)
    rz = -1.60 + rng.normal(0.0, 0.005, n_road).astype(np.float32)
    ri = rng.uniform(0.15, 0.35, n_road).astype(np.float32)

    # 2. Curbs & Sidewalks
    n_sw = 1500
    sw_x = rng.uniform(-15.0, 35.0, n_sw).astype(np.float32)
    sw_side = rng.choice([-1.0, 1.0], n_sw).astype(np.float32)
    sw_y = sw_side * rng.uniform(4.0, 7.5, n_sw).astype(np.float32)
    sw_z = -1.50 + rng.normal(0.0, 0.006, n_sw).astype(np.float32)
    sw_i = rng.uniform(0.20, 0.40, n_sw).astype(np.float32)

    # 3. Dynamic Moving Vehicle (cruising forward along x at speed_mps = 3.5 m/s)
    t = frame_idx * dt
    veh_center_x = float(5.0 + 3.5 * t)
    veh_center_y = float(1.8)
    veh_center_z = float(-0.9)

    n_veh = 800
    vl, vw, vh = 4.2, 1.8, 1.4
    vx = rng.uniform(veh_center_x - vl * 0.5, veh_center_x + vl * 0.5, n_veh).astype(np.float32)
    vy = rng.uniform(veh_center_y - vw * 0.5, veh_center_y + vw * 0.5, n_veh).astype(np.float32)
    vz = rng.uniform(veh_center_z - vh * 0.5, veh_center_z + vh * 0.5, n_veh).astype(np.float32)
    vi = rng.uniform(0.70, 0.90, n_veh).astype(np.float32)

    # 4. Static Obstacles
    n_obs = 300
    ox = rng.normal(14.0, 0.1, n_obs).astype(np.float32)
    oy = rng.normal(-5.2, 0.1, n_obs).astype(np.float32)
    oz = rng.uniform(-1.5, 1.8, n_obs).astype(np.float32)
    oi = rng.uniform(0.4, 0.6, n_obs).astype(np.float32)

    all_x = np.concatenate([rx, sw_x, vx, ox])
    all_y = np.concatenate([ry, sw_y, vy, oy])
    all_z = np.concatenate([rz, sw_z, vz, oz])
    all_i = np.concatenate([ri, sw_i, vi, oi])
    total_pts = len(all_x)
    ring = (np.arange(total_pts) % 64).astype(np.float32)

    return np.column_stack([all_x, all_y, all_z, all_i, ring]).astype(np.float32)


@pytest.fixture
def sample_sweep_points():
    """Provides a single in-memory procedural sweep [N, 5]."""
    return generate_procedural_sweep(0)

