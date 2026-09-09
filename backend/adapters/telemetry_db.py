import queue
import sqlite3
import threading
import time
from typing import Any, Dict, List

class AsyncTelemetryDB:
    def __init__(self, db_path: str = "mission_analytics.db"):
        self.db_path = db_path
        self.log_queue = queue.Queue()
        self.running = True
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
                    ram_usage_mb REAL NOT NULL,
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
            # ISSUE 1 FIXED: Added the mandatory hazard_events schema
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

    def log_health(self, fps: float, latency: float, cells: int, ram_mb: float, vram_mb: float = 0.0):
        self.log_queue.put(("health", (time.time(), fps, latency, cells, ram_mb, vram_mb)))

    def log_tracks(self, tracks: List[Dict[str, Any]]):
        now = time.time()
        for t in tracks:
            pos = t.get("position", [t.get("x", 0.0), t.get("y", 0.0)])
            vel = t.get("velocity", [t.get("vx", 0.0), t.get("vy", 0.0)])
            data = (
                now,
                t.get("id", 0),
                t.get("class", "unknown"),
                pos[0],
                pos[1],
                vel[0],
                vel[1],
                t.get("speed", 0.0),
                t.get("heading", 0.0)
            )
            self.log_queue.put(("track", data))

    # ISSUE 1 FIXED: Added method to log tactical safety interventions
    def log_hazard(self, hazard_type: str, loc_x: float, loc_y: float, margin: float, action: str):
        self.log_queue.put(("hazard", (time.time(), hazard_type, loc_x, loc_y, margin, action)))

    def _writer_loop(self):
        conn = sqlite3.connect(self.db_path)
        # ISSUE 2 FIXED: Enable Write-Ahead Logging for high-speed concurrent writes
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        cursor = conn.cursor()
        
        while self.running:
            batch_processed = False
            try:
                # Block for up to 1.0 second waiting for data
                item = self.log_queue.get(timeout=1.0)
                
                # ISSUE 2 FIXED: Drain the queue rapidly before committing to disk
                while True:
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
                    
                    batch_processed = True
                    # Grab next item instantly. If empty, throws queue.Empty
                    item = self.log_queue.get_nowait()
                    
            except queue.Empty:
                # Commit the entire batch exactly once per second
                if batch_processed:
                    conn.commit()
                    
        # Final cleanup commit
        conn.commit()
        conn.close()