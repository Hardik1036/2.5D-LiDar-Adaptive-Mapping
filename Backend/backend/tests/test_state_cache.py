from unittest.mock import MagicMock
import pytest
import redis.exceptions
import time
from backend.telemetry.state_cache import StateCache


@pytest.fixture
def cache():
    # Force memory mode for 100% isolated, deterministic testing without localhost Redis dependencies
    return StateCache(force_memory_mode=True)


def test_serialization_deserialization(cache):
    pose = (1.5, 2.5, 3.5, 1.0, 15.0)
    tracks = [{"id": 42, "class": "pedestrian", "x": 10.0, "y": 12.0, "vx": 0.5, "vy": 0.5, "speed": 0.7, "heading": 0.78}]
    bounds = {"x_min": -50.0, "x_max": 50.0, "y_min": -50.0, "y_max": 50.0, "depth": 5, "cell_count": 64}

    assert cache.update_pose(*pose)
    assert cache.update_active_tracks(tracks)
    assert cache.update_quadtree_bounds(bounds)

    payload = cache.get_broadcast_payload()

    assert payload["pose"] == {"x": 1.5, "y": 2.5, "z": 3.5, "yaw": 1.0, "speed": 15.0}
    assert payload["tracks"] == tracks
    assert payload["bounds"] == bounds


def test_benchmark_latency(cache):
    iterations = 200
    start_time = time.time()

    pose = (1.5, 2.5, 3.5, 1.0, 15.0)
    tracks = [{"id": 42, "class": "pedestrian", "x": 10.0, "y": 12.0, "vx": 0.5, "vy": 0.5, "speed": 0.7, "heading": 0.78}]
    bounds = {"x_min": -50.0, "x_max": 50.0, "y_min": -50.0, "y_max": 50.0, "depth": 5, "cell_count": 64}

    for _ in range(iterations):
        cache.update_pose(*pose)
        cache.update_active_tracks(tracks)
        cache.update_quadtree_bounds(bounds)
        cache.get_broadcast_payload()

    end_time = time.time()
    avg_latency_ms = ((end_time - start_time) * 1000) / iterations

    assert avg_latency_ms < 0.8, f"Benchmark failed: {avg_latency_ms:.4f} ms exceeds 0.8 ms SLA"


def test_runtime_redis_disconnection_fallback():
    """Verifies seamless fallback to in-memory mode when Redis drops mid-flight."""
    c = StateCache(force_memory_mode=True)
    c.mode = 'redis'
    c.use_redis = True

    # Mock a Redis client that raises ConnectionError during write
    mock_client = MagicMock()
    mock_client.set.side_effect = redis.exceptions.ConnectionError("Connection dropped mid-flight")
    c.redis_client = mock_client

    # Write should not crash; it must log warning and fall back to local memory
    success = c.update_pose(1.0, 2.0, 3.0, 0.0, 5.0)
    assert success is True
    assert c.mode == 'memory'
    assert c.use_redis is False

    # Verify payload was safely written to in-memory store
    payload = c.get_broadcast_payload()
    assert payload["pose"] == {"x": 1.0, "y": 2.0, "z": 3.0, "yaw": 0.0, "speed": 5.0}

    # Subsequent reads and writes work seamlessly in memory mode
    assert c.update_active_tracks([{"id": 99}]) is True
    payload2 = c.get_broadcast_payload()
    assert payload2["tracks"] == [{"id": 99}]
