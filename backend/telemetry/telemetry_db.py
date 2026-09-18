import os
import queue
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

from backend.config import CONFIG


class AsyncTelemetryDB:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            self.db_path = os.path.join(os.path.dirname(__file__), "mission_analytics.db")
        else:
            self.db_path = db_path
        self.log_queue = queue.Queue()
        self.running = True
        self._closed = False
        self._init_db()
        self.worker = threading.Thread(target=self._writer_loop, daemon=True)
        self.worker.start()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    loop_fps REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    active_quadtree_cells INTEGER,
                    ram_usage_mb REAL,
                    vram_usage_mb REAL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dynamic_tracks_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    track_id INTEGER NOT NULL,
                    object_class TEXT NOT NULL,
                    pos_x REAL NOT NULL,
                    pos_y REAL NOT NULL,
                    vel_x REAL NOT NULL,
                    vel_y REAL NOT NULL,
                    speed REAL NOT NULL,
                    heading REAL NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hazard_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    hazard_type TEXT NOT NULL,
                    location_x REAL NOT NULL,
                    location_y REAL NOT NULL,
                    clearance_margin REAL,
                    action_taken TEXT NOT NULL
                )
            """)
            conn.commit()

    def log_health(
        self,
        fps: float,
        latency: float,
        cells: int,
        ram_mb: Optional[float] = None,
        vram_mb: Optional[float] = None,
    ):
        if not self._closed:
            self.log_queue.put(("health", (time.time(), fps, latency, cells, ram_mb, vram_mb)))

    def log_tracks(self, tracks: Any):
        if self._closed or not tracks:
            return
        now = time.time()
        for t in tracks:
            if isinstance(t, dict):
                pos = t.get("position", [t.get("x", 0.0), t.get("y", 0.0)])
                vel = t.get("velocity", [t.get("vx", 0.0), t.get("vy", 0.0)])
                track_id = t.get("id", t.get("track_id", 0))
                cls_name = t.get("class", "unknown")
                x = pos[0] if len(pos) > 0 else 0.0
                y = pos[1] if len(pos) > 1 else 0.0
                vx = vel[0] if len(vel) > 0 else 0.0
                vy = vel[1] if len(vel) > 1 else 0.0
                speed = t.get("speed", 0.0)
                heading = t.get("heading", 0.0)
            else:
                track_id = getattr(t, "track_id", getattr(t, "id", 0))
                cls_name = getattr(t, "classification", "unknown")
                x = getattr(t, "x", 0.0)
                y = getattr(t, "y", 0.0)
                vx = getattr(t, "vx", 0.0)
                vy = getattr(t, "vy", 0.0)
                speed = getattr(t, "speed", 0.0)
                heading = getattr(t, "heading", 0.0)
            data = (
                now,
                track_id,
                cls_name,
                x,
                y,
                vx,
                vy,
                speed,
                heading,
            )
            self.log_queue.put(("track", data))

    def log_hazard(self, hazard_type: str, loc_x: float, loc_y: float, margin: float, action: str):
        if not self._closed:
            self.log_queue.put(("hazard", (time.time(), hazard_type, loc_x, loc_y, margin, action)))

    def _insert_record(self, cursor: sqlite3.Cursor, item: tuple):
        record_type, data = item
        if record_type == "health":
            cursor.execute("""
                INSERT INTO system_health 
                (timestamp, loop_fps, latency_ms, active_quadtree_cells, ram_usage_mb, vram_usage_mb)
                VALUES (?, ?, ?, ?, ?, ?)
            """, data)
        elif record_type == "track":
            cursor.execute("""
                INSERT INTO dynamic_tracks_history
                (timestamp, track_id, object_class, pos_x, pos_y, vel_x, vel_y, speed, heading)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data)
        elif record_type == "hazard":
            cursor.execute("""
                INSERT INTO hazard_events
                (timestamp, hazard_type, location_x, location_y, clearance_margin, action_taken)
                VALUES (?, ?, ?, ?, ?, ?)
            """, data)

    def _writer_loop(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        cursor = conn.cursor()

        max_batch_size = 100
        batch_count = 0
        last_commit_time = time.time()

        while self.running:
            try:
                item = self.log_queue.get(timeout=1.0)
            except queue.Empty:
                if batch_count > 0:
                    conn.commit()
                    batch_count = 0
                    last_commit_time = time.time()
                continue

            if item is None:
                # Sentinel received: stop receiving new items
                break

            self._insert_record(cursor, item)
            batch_count += 1

            # Drain batch up to bounded maximum size
            while batch_count < max_batch_size:
                try:
                    next_item = self.log_queue.get_nowait()
                except queue.Empty:
                    break

                if next_item is None:
                    self.running = False
                    break

                self._insert_record(cursor, next_item)
                batch_count += 1

            now = time.time()
            if batch_count >= max_batch_size or (batch_count > 0 and (now - last_commit_time) >= 1.0):
                conn.commit()
                batch_count = 0
                last_commit_time = now

        # Drain any remaining queued items before final commit & close
        while True:
            try:
                item = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                self._insert_record(cursor, item)
                batch_count += 1
                if batch_count >= max_batch_size:
                    conn.commit()
                    batch_count = 0

        if batch_count > 0:
            conn.commit()
        conn.close()

    def close(self, timeout: float = 2.0):
        """
        Deterministic shutdown path:
        Sends sentinel to worker thread, waits for termination, drains remaining queue, and closes SQLite.
        """
        if self._closed:
            return
        self._closed = True
        self.running = False
        self.log_queue.put(None)  # Sentinel
        if self.worker.is_alive():
            self.worker.join(timeout=timeout)

    def stop(self, timeout: float = 2.0):
        """Alias for deterministic shutdown."""
        self.close(timeout=timeout)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()