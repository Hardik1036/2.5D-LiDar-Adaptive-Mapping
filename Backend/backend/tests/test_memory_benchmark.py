import os
import sys
import pytest
import numpy as np

from backend.mapping.memory_benchmark import (
    BYTES_PER_CELL,
    PS26053_MEMORY_BENCHMARK_DATASET,
    MemoryBenchmarkResult,
    Uniform5cmGrid,
    compute_theoretical_uniform_baseline,
    compute_theoretical_uniform_baseline_details,
    compute_adaptive_theoretical_memory,
    measure_spatial_index_mapping_memory_mib,
    measure_spatial_index_memory,
    measure_process_rss_mib,
    get_process_rss_mb,
    run_empirical_memory_benchmark,
    run_canonical_memory_benchmark,
    run_secondary_multizone_stress_test,
)
from backend.config import MapBounds, FOVEATED
from backend.mapping.foveated_grid import FoveatedGrid
from backend.adapters.ml_adapter import MLInferenceEngine


class TestMemoryBenchmark:
    """PS 26053 Memory Benchmark & Integrity Regression Tests."""

    def test_full_domain_uniform_theoretical_memory(self):
        """
        Audit Item 1: Full-domain Theoretical Uniform 5 cm Baseline Calculation.
        Domain: 120m x 100m, resolution: 0.05m.
        width_cells = 120 / 0.05 = 2400
        height_cells = 100 / 0.05 = 2000
        total_cells = 4,800,000
        BYTES_PER_CELL = 48
        theoretical_bytes = 4,800,000 * 48 = 230,400,000 bytes
        theoretical_mib = 230,400,000 / 1,048,576 = 219.7265625 MiB (~219.73 MiB).
        """
        details = compute_theoretical_uniform_baseline_details(MapBounds, resolution=0.05, bytes_per_cell=BYTES_PER_CELL)

        assert details["domain_width"] == 120.0, f"Expected domain width 120m, got {details['domain_width']}"
        assert details["domain_height"] == 100.0, f"Expected domain height 100m, got {details['domain_height']}"
        assert details["cell_size"] == 0.05, f"Expected cell size 0.05m, got {details['cell_size']}"
        assert details["width_cells"] == 2400, f"Expected 2400 width cells, got {details['width_cells']}"
        assert details["height_cells"] == 2000, f"Expected 2000 height cells, got {details['height_cells']}"
        assert details["number_of_cells"] == 4800000, f"Expected 4,800,000 cells, got {details['number_of_cells']}"
        assert details["bytes_per_cell"] == 48, f"Expected BYTES_PER_CELL=48, got {details['bytes_per_cell']}"
        assert details["theoretical_bytes"] == 230400000, f"Expected 230,400,000 bytes, got {details['theoretical_bytes']}"

        expected_mib = 230400000.0 / (1024.0 * 1024.0)
        assert abs(details["exact_theoretical_mib"] - expected_mib) < 1e-6
        assert details["theoretical_mib"] == 219.73, f"Expected 219.73 MiB, got {details['theoretical_mib']}"

        baseline_mib = compute_theoretical_uniform_baseline(MapBounds, resolution=0.05, bytes_per_cell=BYTES_PER_CELL)
        assert baseline_mib == 219.73

    def test_canonical_memory_workload(self):
        """
        Audit Item 2: Canonical PS 26053 Memory Benchmark Workload.
        Dataset: procedural_sweep (8,600 points).
        Evaluates uniform 5cm occupied mapping vs adaptive foveated mapping.
        """
        res = run_canonical_memory_benchmark(bounds=MapBounds, repetitions=2, warm_up=True)

        assert res.canonical_workload_name == "procedural_sweep"
        assert res.canonical_workload_point_count == 8600
        assert res.uniform_5cm_empirical_occupied_mapping_memory_mib is not None
        assert res.adaptive_empirical_mapping_memory_mib is not None
        assert res.empirical_memory_reduction_percent is not None

        # Adaptive memory must be strictly less than uniform 5cm occupied mapping memory
        assert res.adaptive_empirical_mapping_memory_mib < res.uniform_5cm_empirical_occupied_mapping_memory_mib

        # Verified under 15 MiB target budget
        assert res.adaptive_empirical_mapping_memory_mib < 15.0
        assert res.mapping_memory_target_status == "PASS"

        # Secondary multi-zone stress test is distinctly labeled
        stress_res = run_secondary_multizone_stress_test(bounds=MapBounds, repetitions=1, warm_up=False)
        assert stress_res["workload_name"] == "secondary_multi_zone_stress_test"
        assert stress_res["point_count"] == 4000
        assert stress_res["secondary_multi_zone_stress_reduction_percent"] is not None

    def test_uniform_and_adaptive_use_same_memory_accounting(self):
        """
        Audit Item 4: Symmetrical Method A Deep Ownership Accounting.
        Both Uniform5cmGrid and FoveatedGrid are evaluated with the identical
        accounting function: measure_spatial_index_mapping_memory_mib.
        Both include slotted node/cell structures, CellStats, dictionary hash tables,
        and auxiliary coordinate buffers.
        """
        rng = np.random.RandomState(42)
        n = 4000
        pts = rng.uniform(-10, 40, (n, 3)).astype(np.float32)

        u_grid = Uniform5cmGrid(bounds=MapBounds)
        u_grid.build_uniform(pts)

        f_grid = FoveatedGrid(bounds=MapBounds)
        f_grid.build_foveated(pts)

        # Both passed through the exact same accounting function
        u_mem = measure_spatial_index_mapping_memory_mib(u_grid)
        a_mem = measure_spatial_index_mapping_memory_mib(f_grid)

        assert u_mem > 0.0
        assert a_mem > 0.0
        assert a_mem < u_mem
        assert u_mem < 15.0
        assert a_mem < 15.0

    def test_empirical_memory_reduction_formula(self):
        """
        Audit Item 3 & 4: Empirical Reduction Formula Verification.
        empirical_memory_reduction_percent = (1 - adaptive_empirical / uniform_empirical) * 100.
        """
        rng = np.random.RandomState(42)
        n = 30000
        pts = rng.uniform(-15, 45, (n, 3)).astype(np.float32)
        res = run_empirical_memory_benchmark(pts, bounds=MapBounds)

        expected_reduction = (
            1.0 - res.adaptive_empirical_mapping_memory_mib / res.uniform_5cm_empirical_occupied_mapping_memory_mib
        ) * 100.0

        assert abs(res.empirical_memory_reduction_percent - round(expected_reduction, 2)) < 0.05

    def test_mapping_memory_scope(self):
        """
        Audit Item 5: Scope of <15 MiB Mapping Memory Target.
        Target (<15 MiB) applies strictly to spatial-index mapping memory.
        Process RSS is reported separately as informational only.
        """
        res = run_canonical_memory_benchmark(bounds=MapBounds, repetitions=1, warm_up=False)

        assert res.mapping_memory_target_mib == 15.0
        assert res.adaptive_empirical_mapping_memory_mib < 15.0
        assert res.mapping_memory_target_status == "PASS"

        assert res.process_rss_scope == "informational only; not mapping memory"
        # Process RSS encompasses the Python interpreter, runtime, etc.
        # It must be reported separately and is strictly greater than mapping memory
        assert res.process_rss_mib > res.adaptive_empirical_mapping_memory_mib

    def test_performance_target_not_met_status(self):
        """
        Audit Item 6: Honest Performance Limitation Reporting on Current CPU.
        performance_measurement = VERIFIED
        performance_target = NOT_MET_ON_CURRENT_CPU
        Note suggests GPU/CUDA/TensorRT acceleration as next optimisation path.
        """
        from backend.main import DrishtiEngine
        engine = DrishtiEngine(enable_websocket=False)

        rng = np.random.RandomState(42)
        pts = rng.uniform(-10, 30, (500, 4)).astype(np.float32)
        frame_payload = engine.process_frame(pts, frame_id=1, timestamp=0.0)

        sys_stats = frame_payload.get("system_stats", {})
        perf_meas = sys_stats.get("performance_measurement", {})
        perf_target = sys_stats.get("performance_target", {})

        assert perf_meas.get("status") == "VERIFIED"
        assert perf_target.get("status") == "NOT_MET_ON_CURRENT_CPU"
        assert "GPU/CUDA/TensorRT acceleration should be benchmarked" in perf_target.get("note", "")
        assert perf_target.get("target_latency_ms") == 30.0
        assert perf_target.get("target_fps") == 33.0

    def test_pointpillars_unavailable_status(self):
        """
        Audit Item 7: PointPillars Honest Status.
        pointpillars_status = NOT_AVAILABLE
        3d_detection_backend = GEOMETRIC_FALLBACK
        """
        engine = MLInferenceEngine()
        status = engine.get_model_status()

        assert status["pointpillars_status"] in ("NOT_AVAILABLE", "AVAILABLE / DISABLED")
        assert status["3d_detection_backend"] == "GEOMETRIC_FALLBACK"
        assert status["3d_object_detection"]["status"] in ("NOT_AVAILABLE", "OPTIONAL / DISABLED")

    def test_geometric_fallback_status(self):
        """
        Audit Item 7: Geometric Fallback Status.
        geometric_fallback_status = ACTIVE
        """
        engine = MLInferenceEngine()
        status = engine.get_model_status()

        assert status["geometric_fallback_status"] == "ACTIVE"
        assert status["geometric_detection_fallback"]["status"] == "ACTIVE"
        assert status["geometric_detection_fallback"]["backend"] == "GEOMETRIC_FALLBACK"

    def test_tracking_backend_identity(self):
        """
        Audit Item 8: Tracking Backend Identity.
        Linear constant-velocity Kalman tracking with Hungarian data association on geometric fallback detections.
        """
        engine = MLInferenceEngine()
        status = engine.get_model_status()

        assert status["tracking_backend"] == "Linear constant-velocity Kalman tracking with Hungarian data association on geometric fallback detections"
        assert status["tracking_on_geometric_detections"] is True

    def test_model_architecture_source(self):
        """
        Audit Item 9: Model Architecture Source and File Size Separation.
        Architecture declared by model_config.json: LightweightSalsaNext.
        model_file_size_mib = 7.36
        model_runtime_memory_mib = NOT_ISOLATED
        """
        engine = MLInferenceEngine()
        status = engine.get_model_status()

        if status["segmentation"] == "LOADED":
            assert status["model_architecture_source"] == "Declared by model_config.json: LightweightSalsaNext"
            assert status["model_file_size_mib"] == 7.36
            assert status["model_runtime_memory_mib"] == "NOT_ISOLATED"

            onnx_path = os.path.join(engine.model_dir, "SegmentationModel.onnx")
            actual_file_mib = round(os.path.getsize(onnx_path) / (1024.0 * 1024.0), 2)
            assert actual_file_mib == 7.36

    def test_benchmark_reporting_integrity(self):
        """
        Audit Item 10: Benchmark Reporting Integrity.
        Zero fabrication: unavailable ML accuracy is reported as null with GROUND TRUTH NOT AVAILABLE.
        """
        engine = MLInferenceEngine()
        assert engine.accuracy is None
        assert engine.ground_truth_status == "GROUND TRUTH NOT AVAILABLE"

        status = engine.get_model_status()
        assert "available_onnxruntime_providers" in status
        assert "CPUExecutionProvider" in status["available_onnxruntime_providers"]

        # Backward compatibility check for aliases
        details = compute_theoretical_uniform_baseline_details(MapBounds)
        assert details["theoretical_mib"] == 219.73
