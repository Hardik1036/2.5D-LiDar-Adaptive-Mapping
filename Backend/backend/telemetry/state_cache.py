import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional
import redis
import redis.exceptions

logger = logging.getLogger("StateCache")


class StateCache:
    """
    High-speed telemetry state cache supporting Redis with deterministic
    in-memory fallback and seamless mid-flight disconnection recovery.
    """
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        force_memory_mode: bool = False,
    ):
        # 1. Unconditionally initialize thread lock and in-memory store
        self._lock = threading.Lock()
        self._memory_store: Dict[str, Optional[str]] = {
            'vehicle:pose': None,
            'tracks:active': None,
            'quadtree:bounds': None,
        }
        self._cache = self._memory_store  # Backwards compatibility alias

        if force_memory_mode:
            self.mode = 'memory'
            self.use_redis = False
            self.redis_client = None
            return

        self.mode = 'redis'
        self.use_redis = True
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            socket_timeout=0.1,
            socket_connect_timeout=0.1,
        )

        try:
            self.redis_client.ping()
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError, Exception) as e:
            logger.warning(f"Redis connection failed: {e}. Falling back to in-memory mode.")
            self.mode = 'memory'
            self.use_redis = False

    def update_pose(self, x: float, y: float, z: float, yaw: float, speed: float) -> bool:
        data = {"x": x, "y": y, "z": z, "yaw": yaw, "speed": speed}
        payload = json.dumps(data)

        if self.use_redis and self.redis_client is not None:
            try:
                self.redis_client.set('vehicle:pose', payload)
                with self._lock:
                    self._memory_store['vehicle:pose'] = payload
                return True
            except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError, Exception) as exc:
                logger.warning(f"Redis write error ({exc}). Disconnecting and falling back to in-memory mode.")
                self.mode = 'memory'
                self.use_redis = False

        with self._lock:
            self._memory_store['vehicle:pose'] = payload
        return True

    def update_active_tracks(self, tracks_list: list) -> bool:
        payload = json.dumps(tracks_list)

        if self.use_redis and self.redis_client is not None:
            try:
                self.redis_client.set('tracks:active', payload)
                with self._lock:
                    self._memory_store['tracks:active'] = payload
                return True
            except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError, Exception) as exc:
                logger.warning(f"Redis write error ({exc}). Disconnecting and falling back to in-memory mode.")
                self.mode = 'memory'
                self.use_redis = False

        with self._lock:
            self._memory_store['tracks:active'] = payload
        return True

    def update_quadtree_bounds(self, bounds_dict: dict) -> bool:
        payload = json.dumps(bounds_dict)

        if self.use_redis and self.redis_client is not None:
            try:
                self.redis_client.set('quadtree:bounds', payload)
                with self._lock:
                    self._memory_store['quadtree:bounds'] = payload
                return True
            except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError, Exception) as exc:
                logger.warning(f"Redis write error ({exc}). Disconnecting and falling back to in-memory mode.")
                self.mode = 'memory'
                self.use_redis = False

        with self._lock:
            self._memory_store['quadtree:bounds'] = payload
        return True

    def get_broadcast_payload(self) -> dict:
        result: Dict[str, Any] = {
            "pose": None,
            "tracks": [],
            "bounds": None,
        }

        if self.use_redis and self.redis_client is not None:
            try:
                pipe = self.redis_client.pipeline()
                pipe.get('vehicle:pose')
                pipe.get('tracks:active')
                pipe.get('quadtree:bounds')
                pose_raw, tracks_raw, bounds_raw = pipe.execute()

                if pose_raw:
                    result['pose'] = json.loads(pose_raw)
                if tracks_raw:
                    result['tracks'] = json.loads(tracks_raw)
                if bounds_raw:
                    result['bounds'] = json.loads(bounds_raw)
                return result
            except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError, Exception) as exc:
                logger.warning(f"Redis read error ({exc}). Disconnecting and falling back to in-memory mode.")
                self.mode = 'memory'
                self.use_redis = False

        with self._lock:
            pose_raw = self._memory_store.get('vehicle:pose')
            tracks_raw = self._memory_store.get('tracks:active')
            bounds_raw = self._memory_store.get('quadtree:bounds')

            if pose_raw:
                result['pose'] = json.loads(pose_raw)
            if tracks_raw:
                result['tracks'] = json.loads(tracks_raw)
            if bounds_raw:
                result['bounds'] = json.loads(bounds_raw)

        return result


if __name__ == "__main__":
    cache = StateCache(force_memory_mode=True)
    start_time = time.time()
    iterations = 1000

    for _ in range(iterations):
        cache.update_pose(1.0, 2.0, 3.0, 0.5, 10.0)
        cache.update_active_tracks([{"id": 1, "class": "car", "x": 5.0, "y": 5.0, "vx": 1.0, "vy": 0.0, "speed": 1.0, "heading": 0.0}])
        cache.update_quadtree_bounds({"x_min": 0.0, "x_max": 10.0, "y_min": 0.0, "y_max": 10.0, "depth": 3, "cell_count": 8})
        payload = cache.get_broadcast_payload()

    end_time = time.time()
    total_time_ms = (end_time - start_time) * 1000
    avg_latency_ms = total_time_ms / iterations

    print(f"Executed {iterations} write/read cycles in {total_time_ms:.2f} ms")
    print(f"Average operation latency: {avg_latency_ms:.4f} ms")

    # Assert SLA
    assert avg_latency_ms < 0.8, f"SLA Violation: Average latency {avg_latency_ms:.4f} ms >= 0.8 ms"
    print("SLA Self-Test Passed: Latency < 0.8 ms")
