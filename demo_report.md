# DRDO SIH 2026 Problem Statement 53: Evaluation & Benchmark Report

**Project:** Adaptive Variable-Resolution 2.5D LiDAR Perception Engine  
**Test Date:** 2026-09-06 14:29:26  
**Evaluation Scope:** 120 Continuous Multi-Phase Tactical Sweeps (Headless Zero-Dependency Execution)

---

## 1. Executive Performance Verdict

| Metric | Measured Result | DRDO Target SLA | Status |
| :--- | :---: | :---: | :---: |
| **Real-Time Throughput** | **130.53 Hz** | $\ge 25.0\text{ Hz}$ | **PASS** |
| **Mean Loop Latency** | **11.01 ms** | $\le 35.0\text{ ms}$ | **PASS** |
| **P95 Frame Latency** | **18.17 ms** | $\le 40.0\text{ ms}$ | **PASS** |
| **Memory Footprint** | **0.012 MB** | $\le 16.0\text{ MB}$ | **PASS** |
| **RAM Compression** | **99.94%** | $\ge 70 - 78\%$ | **PASS (Exceeds Target)** |

---

## 2. Multi-Phase Tactical Verification Results

| Phase | Operational Conditions | Observed Behavior & Verification | Mean Latency |
| :--- | :--- | :--- | :---: |
| **Phase 1 (Frames 1–30)** | Flat open ground | Pure coarse 50 cm grid; zero fine subdivision; zero false obstacles. | **4.34 ms** |
| **Phase 2 (Frames 31–60)** | Approaching boulder & trench | Dynamic 5 cm subdivision triggered on boulder; Trench detector stamps drop-off edge as lethal ($255$). | **16.24 ms** |
| **Phase 3 (Frames 61–90)** | Pedestrian crossing at 1.4 m/s | 2D EKF tracks obstacle heading & speed; forward hazard cones inflate safe clearance; ghost trails cleared. | **9.77 ms** |
| **Phase 4 (Frames 91–120)** | Dust burst & tall grass patch | Statistical k-NN filter eliminates airborne scatter; Porosity classifier caps tall grass traversability cost at $\le 20$. | **13.67 ms** |

---

## 3. Latency Distribution Breakdown

- **Minimum Latency:** `2.62 ms`
- **Median (P50) Latency:** `11.80 ms`
- **95th Percentile (P95):** `18.17 ms`
- **99th Percentile (P99):** `21.62 ms`
- **Maximum Latency:** `22.43 ms`

---

## 4. Architectural Innovations Proven

1. **Strict 2.5D Ground Plane Statistics:** Zero 3D voxels or Octrees allocated, eliminating GPU requirements and frame-to-frame garbage collection pauses.
2. **Deterministic Dual-Resolution Memory:** Dynamic coarse ($50\text{ cm}$) to fine ($5\text{ cm}$) spatial refinement preserves centimeter-accurate edge boundaries while saving over 99% RAM.
3. **Tactical Multi-Modal Filtering:** Combines k-NN atmospheric outlier removal, laser penetration porosity analysis, and millimeter-residual ground hazard detection into a unified sub-35ms pipeline.

**Conclusion:** All DRDO operational constraints, edge-compute latency bounds, and memory reduction targets are deterministically met.
