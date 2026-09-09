"""
Redis State Synchronization Adapter with In-Memory Offline Fallback.
Manages vehicle odometry pose caching, dynamic track synchronization,
and telemetry streaming for DRDO SIH 2026 Problem Statement 53.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from backend.config import DATABASE, DatabaseConfig
from backend.adapters.telemetry_db import AsyncTelemetryDB

logger = logging.getLogger(__name__)


class DatabaseAdapter:
    """
    High-speed key-value state adapter interfacing with Redis.
    Guarantees deterministic, non-blocking fallback to internal in-memory storage
    when Redis is offline or latency exceeds the 50ms socket timeout SLA.
    """

    def __init__(self, config: DatabaseConfig = DATABASE, enabled: bool = True):
        self.config = config
        self.enabled = enabled
        self.client = None
        self._is_connected = False
        
        self.sqlite_db = AsyncTelemetryDB()
        
        self._memory_store: Dict[str, Any] = {
            self.config.KEY_POSE: json.dumps({"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}),
            self.config.KEY_TRACKS: json.dumps([]),
            self.config.KEY_TELEMETRY: [],
        }

        if self.enabled:
            self._connect()

    def _connect(self):
        """Attempts connection to Redis with short socket timeout."""
        try:
            # Load Redis lazily so the adapter remains usable when the optional
            # Redis client dependency is not installed.
            redis = __import__("redis")
            self.client = redis.Redis(
                host=self.config.REDIS_HOST,
                port=self.config.REDIS_PORT,
                db=self.config.REDIS_DB,
                socket_timeout=self.config.SOCKET_TIMEOUT,
                socket_connect_timeout=self.config.SOCKET_TIMEOUT,
                decode_responses=True,
            )
            # Fast ping check
            self.client.ping()
            self._is_connected = True
            logger.info("Connected to Redis cache at %s:%s", self.config.REDIS_HOST, self.config.REDIS_PORT)
        except Exception as e:
            self._is_connected = False
            self.client = None
            logger.warning(
                "Redis unavailable (%s). Falling back to non-blocking in-memory state store.", e
            )

    @property
    def is_connected(self) -> bool:
        """Indicates whether active Redis server connection is established."""
        return self._is_connected

    def get_vehicle_pose(self) -> Tuple[float, float, float, float]:
        """
        Retrieves current vehicle pose (x, y, z, yaw).
        Returns (0.0, 0.0, 0.0, 0.0) if unavailable.
        """
        raw_val = None
        if self._is_connected and self.client is not None:
            try:
                raw_val = self.client.get(self.config.KEY_POSE)
            except Exception as e:
                logger.debug("Redis read error for %s: %s", self.config.KEY_POSE, e)
                raw_val = None

        if raw_val is None:
            raw_val = self._memory_store.get(self.config.KEY_POSE)

        if raw_val:
            try:
                data = json.loads(raw_val) if isinstance(raw_val, str) else raw_val
                return (
                    float(data.get("x", 0.0)),
                    float(data.get("y", 0.0)),
                    float(data.get("z", 0.0)),
                    float(data.get("yaw", 0.0)),
                )
            except Exception:
                pass

        return (0.0, 0.0, 0.0, 0.0)

    def set_vehicle_pose(self, x: float, y: float, z: float, yaw: float) -> bool:
        """Stores updated vehicle pose in Redis and local memory store."""
        payload = json.dumps({"x": round(x, 4), "y": round(y, 4), "z": round(z, 4), "yaw": round(yaw, 4)})
        self._memory_store[self.config.KEY_POSE] = payload

        if self._is_connected and self.client is not None:
            try:
                self.client.set(self.config.KEY_POSE, payload)
                return True
            except Exception as e:
                logger.debug("Redis write error for %s: %s", self.config.KEY_POSE, e)
                return False
        return True

    def sync_active_tracks(self, tracks: List[Any]) -> bool:
        """
        Synchronizes active EKF dynamic tracks with state database.
        Accepts TrackedObstacle instances or raw dictionary records.
        """
        track_list = []
        for t in tracks:
            if hasattr(t, "to_dict"):
                track_list.append(t.to_dict())
            elif isinstance(t, dict):
                track_list.append(t)

        payload = json.dumps(track_list)
        self._memory_store[self.config.KEY_TRACKS] = payload

        # Log to SQLite FIRST to avoid being skipped by the Redis returns below.
        # Some telemetry DB implementations do not expose a log_tracks method,
        # so guard the call to keep compatibility with both interfaces.
        log_tracks = getattr(self.sqlite_db, "log_tracks", None)
        if callable(log_tracks):
            log_tracks(track_list)

        if self._is_connected and self.client is not None:
            try:
                self.client.set(self.config.KEY_TRACKS, payload)
                return True
            except Exception as e:
                logger.debug("Redis write error for %s: %s", self.config.KEY_TRACKS, e)
                return False
        
        return True

    def get_active_tracks(self) -> List[Dict[str, Any]]:
        """Retrieves active tracks from database or local memory cache."""
        raw_val = None
        if self._is_connected and self.client is not None:
            try:
                raw_val = self.client.get(self.config.KEY_TRACKS)
            except Exception:
                raw_val = None

        if raw_val is None:
            raw_val = self._memory_store.get(self.config.KEY_TRACKS)

        if raw_val:
            try:
                return json.loads(raw_val) if isinstance(raw_val, str) else raw_val
            except Exception:
                pass
        return []

    def log_frame_telemetry(self, stats: Dict[str, Any]) -> bool:
        """
        Logs frame compute latency, FPS, and cell statistics to telemetry stream.
        Maintains a rolling ring buffer capped at 100 entries.
        """
        payload = json.dumps(stats)
        
        # Proper indentation fixed here
        self.sqlite_db.log_health(
            fps=stats.get("fps", 0.0),
            latency=stats.get("latency_ms", 0.0),
            cells=stats.get("cell_count", 0),
            ram_mb=(stats.get("cell_count", 0) * 64) / (1024.0 * 1024.0)
        )
        
        # Local memory ring buffer
        mem_stream = self._memory_store.setdefault(self.config.KEY_TELEMETRY, [])
        mem_stream.append(payload)
        if len(mem_stream) > 100:
            del mem_stream[:-100]

        if self._is_connected and self.client is not None:
            try:
                # Use capped list lpush + ltrim
                pipeline = self.client.pipeline()
                pipeline.lpush(self.config.KEY_TELEMETRY, payload)
                pipeline.ltrim(self.config.KEY_TELEMETRY, 0, 99)
                pipeline.execute()
                return True
            except Exception as e:
                logger.debug("Redis stream error: %s", e)
                return False
        return True