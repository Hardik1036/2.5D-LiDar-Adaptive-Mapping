"""
Unit tests for AsyncTelemetryDB (backend/telemetry/telemetry_db.py)
Verifies deterministic shutdown, queue draining, context manager, bounded batches,
and nullable RAM/VRAM handling.
"""

import os
import sqlite3
import tempfile
import time
import pytest

from backend.telemetry.telemetry_db import AsyncTelemetryDB


@pytest.fixture
def temp_db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def test_telemetry_db_context_manager(temp_db_path):
    with AsyncTelemetryDB(db_path=temp_db_path) as db:
        db.log_health(fps=25.0, latency=12.5, cells=500, ram_mb=128.4, vram_mb=None)
        db.log_hazard("thin_hazard", 2.0, 5.0, 0.25, "brake")

    # After exiting context, DB should be closed and records drained
    with sqlite3.connect(temp_db_path) as conn:
        c = conn.cursor()
        c.execute("SELECT loop_fps, latency_ms, active_quadtree_cells, ram_usage_mb, vram_usage_mb FROM system_health")
        health_row = c.fetchone()
        assert health_row is not None
        assert health_row[0] == 25.0
        assert health_row[1] == 12.5
        assert health_row[2] == 500
        assert pytest.approx(health_row[3], rel=1e-3) == 128.4
        assert health_row[4] is None

        c.execute("SELECT hazard_type, location_x, location_y, clearance_margin, action_taken FROM hazard_events")
        hazard_row = c.fetchone()
        assert hazard_row is not None
        assert hazard_row[0] == "thin_hazard"
        assert hazard_row[1] == 2.0
        assert hazard_row[2] == 5.0


def test_deterministic_shutdown_drains_queue(temp_db_path):
    db = AsyncTelemetryDB(db_path=temp_db_path)

    # Queue 150 items to exceed max_batch_size (100)
    for i in range(150):
        db.log_health(fps=float(i), latency=10.0, cells=i, ram_mb=None, vram_mb=None)

    tracks = [
        {"id": 1, "class": "car", "position": [1.0, 2.0], "velocity": [0.5, 0.0], "speed": 0.5, "heading": 0.0},
        {"id": 2, "class": "pedestrian", "position": [3.0, 4.0], "velocity": [0.0, 1.4], "speed": 1.4, "heading": 1.57},
    ]
    db.log_tracks(tracks)

    # Trigger deterministic shutdown immediately
    db.close(timeout=3.0)

    # Confirm worker thread terminated
    assert not db.worker.is_alive()
    assert db._closed

    # Verify all 150 health records and 2 track records persisted
    with sqlite3.connect(temp_db_path) as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM system_health")
        health_count = c.fetchone()[0]
        assert health_count == 150

        c.execute("SELECT COUNT(*) FROM dynamic_tracks_history")
        track_count = c.fetchone()[0]
        assert track_count == 2


def test_nullable_ram_vram_metrics(temp_db_path):
    with AsyncTelemetryDB(db_path=temp_db_path) as db:
        # None passed for both RAM and VRAM
        db.log_health(fps=30.0, latency=8.0, cells=1200, ram_mb=None, vram_mb=None)

    with sqlite3.connect(temp_db_path) as conn:
        c = conn.cursor()
        c.execute("SELECT ram_usage_mb, vram_usage_mb FROM system_health")
        row = c.fetchone()
        assert row is not None
        assert row[0] is None  # ram_usage_mb is NULL
        assert row[1] is None  # vram_usage_mb is NULL
