# Adaptive Variable Resolution 2.5D LiDAR Mapping for Dynamic Perception
### SIH 2026 Problem Statement 53 (DRDO)

An edge-optimized Python 3.10+ real-time perception backend providing 2.5D elevation mapping, adaptive quadtree ground partitioning, dynamic obstacle tracking, expanding hazard cone rollouts, and high-frequency WebSocket telemetry broadcasting.

---

## Key Features

1. **Strict 2.5D Ground Plane Mapping**:
   - Operates on $(X, Y)$ ground planes with compact numerical elevation stats: $[Z_{\min}, Z_{\max}, \Delta Z, \sigma_Z^2, \text{slope}]$.
   - **Zero 3D voxels or Octrees** allocated, ensuring minimal memory footprint and suitability for edge compute platforms.
2. **Adaptive Variable-Resolution Quadtree**:
   - Coarse root resolution: **0.50 m** (flat, safe ground).
   - Fine leaf resolution: **0.05 m** (5 cm).
   - Dynamic subdivision triggered only when step height $\Delta Z > 0.12\text{ m}$ or height variance $\sigma_Z^2 > 0.04\text{ m}^2$.
3. **Real-Time Perception Pipeline ($\ge 25\text{ Hz}$, $< 35\text{ ms}$ latency)**:
   - Concentric polar ring ground plane segmentation ($< 5\text{ ms}$).
   - Fast Euclidean clustering using KD-Tree radius graph search ($< 6\text{ ms}$).
   - 2D Constant-Velocity Extended Kalman Filter (EKF) with Hungarian data association (`linear_sum_assignment`).
4. **Dynamic Threat Perception**:
   - Ghost smear clearing via free-space line-of-sight validation.
   - Expanding elliptical hazard cone projections at $t \in [1.0\text{s}, 2.0\text{s}, 3.0\text{s}]$ oriented along the estimated velocity heading.
5. **Traversability Costmap**:
   - Normalizes terrain slope, roughness, and step heights into cost values:
     - `0 - 50`: Safe traversable terrain.
     - `51 - 180`: Caution / steep slope / rough terrain.
     - `181 - 255`: Lethal obstacle / non-traversable.
6. **Frontend Integration**:
   - Asynchronous WebSocket server broadcasting at `ws://localhost:8765`.
   - Optimized JSON payload formatting with active cells, track states, and hazard polygons.

---

## Directory Structure

```
backend/
├── pyproject.toml               # Poetry/pip configuration with dependencies
├── requirements.txt             # numpy, scipy, filterpy, websockets, open3d
├── README.md                    # Setup, execution instructions, and module guide
├── __init__.py
├── config.py                    # Global bounds, thresholds, and parameters
├── main.py                      # Pipeline orchestrator running the real-time loop
├── ingestion/
│   ├── __init__.py
│   ├── dataset_loader.py        # Parses .bin / .pcd files and synthetic stream
│   └── ground_segmentation.py   # Concentric polar ring ground vs non-ground splitting
├── mapping/
│   ├── __init__.py
│   ├── cell_statistics.py       # Compact statistical tuple: [Z_max, Z_min, delta_Z, var, slope]
│   ├── quadtree.py              # 2D recursive Adaptive Quadtree (coarse 50cm -> fine 5cm)
│   └── costmap.py               # Traversability cost mapping (0 safe, 50 caution, 255 lethal)
├── tracking/
│   ├── __init__.py
│   ├── clustering.py            # Euclidean / DBSCAN clustering on non-ground points
│   ├── kalman_tracker.py        # 2D Extended Kalman Filter (x, y, vx, vy, speed, heading)
│   ├── ghost_clearing.py        # Raycasting to clear dynamic smear trails
│   └── trajectory_rollout.py    # 1.0s - 3.0s expanding elliptical hazard cone projection
├── server/
│   ├── __init__.py
│   ├── payload_builder.py       # Serializes cells and dynamic tracks to frontend JSON
│   └── websocket_server.py      # Async WebSocket broadcaster (ws://localhost:8765)
└── utils/
    ├── __init__.py
    └── synthetic_generator.py   # Programmatic point cloud generator with moving obstacles
```

---

## Installation

Ensure Python 3.10+ is installed on your system.

```bash
cd backend
pip install -r requirements.txt
```

---

## Quickstart & Execution

### 1. Run with Built-in Synthetic LiDAR Generator (Default)
If no dataset path is passed, the pipeline runs the built-in synthetic generator simulating ground slopes, curbs, static barriers, walking pedestrians, and dynamic vehicles:

```bash
python -m backend.main
```

### 2. Run with Custom KITTI or NuScenes Binary Point Clouds (`.bin`)
```bash
python -m backend.main --dataset /path/to/kitti/velodyne_points/data/
```

### 3. Run with Point Cloud Data (`.pcd`)
```bash
python -m backend.main --dataset /path/to/pointclouds/
```

### 4. Adjust Loop Frequency and Port
```bash
python -m backend.main --fps 30.0 --port 8765
```

---

## WebSocket Telemetry Protocol (`ws://localhost:8765`)

The server pushes JSON telemetry frames structured as follows:

```json
{
  "timestamp": 1725612345.123,
  "frame_id": 142,
  "system_stats": {
    "fps": 27.8,
    "latency_ms": 18.4,
    "point_count": 21850,
    "cell_count": 1840,
    "coarse_cells": 1280,
    "fine_cells": 560,
    "refinement_ratio": 0.304,
    "active_tracks": 2,
    "dynamic_tracks": 2
  },
  "cells": [
    {
      "x": 4.25,
      "y": -2.75,
      "size": 0.5,
      "cost": 15,
      "z_min": -1.62,
      "z_max": -1.58,
      "delta_z": 0.04,
      "variance": 0.0008,
      "slope": 3.2,
      "pts": 28
    }
  ],
  "dynamic_objects": [
    {
      "id": 1,
      "position": [4.0, -6.4, -0.72],
      "velocity": [0.0, 1.2],
      "speed": 1.2,
      "heading": 1.57,
      "heading_deg": 90.0,
      "dimensions": [0.5, 0.5, 1.75],
      "is_dynamic": true,
      "hazard_cones": [
        {
          "t": 1.0,
          "center": [4.0, -5.2],
          "semi_major": 0.58,
          "semi_minor": 0.45,
          "heading_deg": 90.0,
          "polygon": [[...], [...]]
        }
      ]
    }
  ]
}
```
