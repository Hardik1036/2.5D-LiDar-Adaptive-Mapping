# Adaptive Variable-Resolution 2.5D LiDAR Perception Engine
### SIH 2026 Problem Statement 53 (DRDO: Defence Research and Development Organisation)

An edge-optimized, real-time Python perception engine providing:
- **Strict 2.5D Ground Plane Mapping**: Zero 3D voxels / Octrees allocated, saving > 99% RAM over dense 5 cm grids.
- **Adaptive Variable-Resolution Quadtree**: 50 cm coarse root to 5 cm fine leaf dynamic subdivision.
- **Tactical Terrain Filters**:
  - Statistical k-NN Dust & Atmospheric Scatter Removal ([Feature F1.3])
  - Millimeter-Residual Ground Hazard & Spike Strip Detection ([Feature F1.6])
  - Multi-Echo Laser Penetration & Porosity Classification ([Feature F1.5])
  - Negative Obstacle / Trench Drop-off Detection ([Feature F5.1])
  - Temporal Map Blending & 1D Elevation Kalman Memory ([Feature F2.4])
  - Sensor Confidence & Degraded-Mode Safe Fallback ([Feature F5.3])
- **Dynamic Perception**: 2D EKF tracking, Hungarian data association, ghost smear clearing, and 1–3s expanding hazard cone projections.
- **24/7 Cloud Production Ready**: Dynamic port routing (`$PORT`), Docker containerization, WebSocket keep-alive heartbeats, and graceful signal management.

---

## Performance SLA Benchmarks

| Metric | Measured Result | DRDO Target SLA | Status |
| :--- | :---: | :---: | :---: |
| **Real-Time Throughput** | **29.25 – 33.14 Hz** | $\ge 25.0\text{ Hz}$ | **PASS** |
| **Mean Frame Latency** | **30.17 – 34.19 ms** | $\le 35.0\text{ ms}$ | **PASS** |
| **P95 Frame Latency** | **33.67 – 37.93 ms** | $\le 40.0\text{ ms}$ | **PASS** |
| **Active Memory Footprint** | **0.10 MB** (vs 19.53 MB dense) | $\le 16.0\text{ MB}$ | **PASS** |
| **RAM Compression** | **99.48% Saved** | $\ge 70 - 78\%$ | **PASS (Exceeds Target)** |

---

## Quick Start

### 1. Local Python Setup
```bash
# Create virtual environment (Python 3.10+)
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run perception engine
python -m backend.main --profile
```

### 2. Zero-Dependency Terminal Visualizer & Benchmark Runner
Run the standalone ANSI radar HUD simulation across 4 tactical phases without browser or Redis:
```bash
python -m backend.simulate_demo --speed 1.0 --record-benchmark
```

### 3. Docker Container Deployment (Railway / Render / Cloud)
```bash
# Build Docker container
docker build -t drdo-lidar-backend:latest .

# Run container with dynamic port routing
docker run -p 8765:8765 -e PORT=8765 drdo-lidar-backend:latest
```

---

## Testing & Verification

Run the complete 18-test automated suite:
```bash
python -m pytest backend/tests/ -v
```

Execute the headless 60-frame edge latency profiler:
```bash
python -m backend.benchmarks.edge_profiler 60
```

Execute the memory reduction audit vs dense 5 cm grid:
```bash
python -m backend.benchmarks.memory_validator
```

---

## License & Attribution
Developed for Smart India Hackathon (SIH) 2026 Problem Statement 53 (DRDO).
