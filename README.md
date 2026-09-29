# Drishti 2.5D Adaptive Mapping & Tactical Perception

High-speed, real-time 2.5D LiDAR elevation mapping and tactical perception system for DRDO SIH 2026 PS-53.

---

## Architecture Overview

The system is organized into two isolated sub-systems:

```
e:\Drishti 2.5D Adaptive mapping\
├── run_all.bat          # One-click launcher for both Backend and Frontend
├── run_backend.bat      # Launches Backend perception pipeline (Port 8765)
├── run_frontend.bat     # Launches Frontend React Vite dev server (Port 5173)
├── Backend/             # Python perception pipeline & ML inference
│   ├── backend/         # Core perception package (adapters, mapping, models, tracking)
│   ├── data/            # KITTI dataset sweeps
│   ├── tests/           # Pytest test suite
│   ├── requirements.txt # Python dependencies
│   └── .env             # Backend environment & Kaggle credentials
└── Frontend/            # React + Three.js 2.5D visualizer
    ├── src/             # Application UI and WebGL rendering components
    ├── tests/           # Frontend unit & integration tests
    ├── package.json     # Node.js dependencies
    └── .env             # Frontend WebSocket URL (ws://127.0.0.1:8765)
```

---

## Quick Start

### 1. Launch Everything Together
Double-click or run:
```powershell
.\run_all.bat
```

### 2. Launch Individually
- **Backend:**
  ```powershell
  .\run_backend.bat
  ```
  *(Or: `cd Backend && E:\anaconda\python.exe -m backend.main`)*

- **Frontend:**
  ```powershell
  .\run_frontend.bat
  ```
  *(Or: `cd Frontend && npm run dev`)*

---

## Endpoints

- **Frontend Web UI:** [http://localhost:5173](http://localhost:5173)
- **Backend WebSocket:** `ws://127.0.0.1:8765`
