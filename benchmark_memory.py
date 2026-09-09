"""
DRDO SIH26053 - Module 4.1
Memory Footprint Benchmark (benchmark_memory.py)
Mathematically proves RAM reduction of Adaptive 2.5D Quadtree vs Dense Baseline.
"""

import sys
import numpy as np
from backend.config import BOUNDS, QUADTREE
from backend.mapping.quadtree import AdaptiveQuadtree
from backend.utils.synthetic_generator import SyntheticLiDARGenerator


def run_memory_benchmark():
    span_x = BOUNDS.X_MAX - BOUNDS.X_MIN  # 40.0 m
    span_y = BOUNDS.Y_MAX - BOUNDS.Y_MIN  # 40.0 m
    fine_res = QUADTREE.FINE_RESOLUTION   # 0.05 m (5 cm)
    coarse_res = QUADTREE.COARSE_RESOLUTION  # 0.50 m (50 cm)

    # 1. Theoretical Dense Grid Baseline (40m x 40m at 5 cm resolution)
    dense_cols = int(np.ceil(span_x / fine_res))  # 800
    dense_rows = int(np.ceil(span_y / fine_res))  # 800
    dense_total_cells = dense_cols * dense_rows   # 640,000 cells
    bytes_per_dense_cell = 32                     # 7 float32 stats + 1 int32 cost
    dense_ram_mb = (dense_total_cells * bytes_per_dense_cell) / (1024.0 * 1024.0)

    # 2. Adaptive Quadtree Allocation under Active Lidar Sweep
    generator = SyntheticLiDARGenerator()
    quadtree = AdaptiveQuadtree(bounds=BOUNDS, coarse_res=coarse_res, fine_res=fine_res)

    pts, _ = generator.generate_frame(timestamp=1.2, dt=0.04)
    leaves = quadtree.build(pts)
    active_cell_count = len(leaves)

    node_slot_bytes = 64  # Slotted QuadtreeNode + compact CellStats
    quadtree_ram_mb = (active_cell_count * node_slot_bytes) / (1024.0 * 1024.0)

    # 3. Compute Savings Percentage
    ram_savings_pct = (1.0 - (quadtree_ram_mb / dense_ram_mb)) * 100.0

    print("=" * 80)
    print("DRDO PS-53 AUDIT: ADAPTIVE 2.5D QUADTREE MEMORY REDUCTION")
    print("=" * 80)
    print(f"Operational Area        : {span_x:.1f} m x {span_y:.1f} m ({span_x * span_y:.0f} m²)")
    print(f"Coarse / Fine Resolution: {coarse_res * 100:.0f} cm coarse  -->  {fine_res * 100:.0f} cm fine")
    print("-" * 80)
    print(f"{'Mapping Engine':<32} | {'Active Cells':<14} | {'RAM Footprint':<14} | {'Cell Size'}")
    print("-" * 80)
    print(f"{'Dense Grid Baseline':<32} | {dense_total_cells:<14,d} | {dense_ram_mb:<11.2f} MB | 32 B")
    print(f"{'Adaptive 2.5D Quadtree (Ours)':<32} | {active_cell_count:<14,d} | {quadtree_ram_mb:<11.4f} MB | 64 B")
    print("=" * 80)
    print(f"Total RAM Saved         : {dense_ram_mb - quadtree_ram_mb:.2f} MB")
    print(f"Reported RAM Reduction  : {ram_savings_pct:.2f}% (Playbook Target: >= 70-78%)")
    print("=" * 80)

    passed = ram_savings_pct >= 70.0
    print(f"Evaluation Verdict      : {'PASS [Compliant with DRDO SLA]' if passed else 'FAIL'}\n")
    return passed


if __name__ == "__main__":
    success = run_memory_benchmark()
    if not success:
        sys.exit(1)