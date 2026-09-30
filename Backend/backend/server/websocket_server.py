"""
High-throughput asynchronous WebSocket server for LiDAR perception telemetry.
Cloud-ready with dynamic port binding, concurrent client broadcasting,
and automatic ping-pong heartbeats to keep proxy sockets alive 24/7.
"""

import asyncio
import http
import logging
import os
from typing import Any, Optional, Set
import json
import websockets
import websockets.exceptions

# Suppress websockets internal logging of rejected HTTP probe connections
logging.getLogger("websockets.server").setLevel(logging.CRITICAL)

try:
    import orjson
    HAS_ORJSON = True
    def fast_serialize(payload: Any) -> str:
        # orjson dumps directly to bytes 10x faster than standard json
        if hasattr(payload, "_dict") and isinstance(payload, str):
            return str(payload)
        return orjson.dumps(payload).decode("utf-8")
except ImportError:
    HAS_ORJSON = False
    import json
    def fast_serialize(payload: Any) -> str:
        if hasattr(payload, "_dict") and isinstance(payload, str):
            return str(payload)
        return json.dumps(payload)


from backend.config import CONFIG, SERVER

logger = logging.getLogger("WebSocketServer")


async def process_request(connection, request=None, *args, **kwargs):
    """
    Handle plain HTTP health-check pings from Railway/Render reverse proxies
    without crashing the WebSocket protocol state machine.
    Compatible with websockets >= 13 (connection, request) and < 13 (path, headers).
    """
    # websockets >= 13: connection, request
    if request is not None and hasattr(request, "headers"):
        if request.headers.get("Upgrade", "").lower() != "websocket":
            if hasattr(connection, "respond"):
                return connection.respond(
                    http.HTTPStatus.OK,
                    "OK\nDRISHTI-2.5D Perception Engine Online\n",
                )
            return (
                http.HTTPStatus.OK,
                [("Content-Type", "text/plain"), ("Content-Length", "42")],
                b"OK\nDRISHTI-2.5D Perception Engine Online\n",
            )
        return None

    # websockets < 13: (path, headers) where connection is path and request is headers
    if isinstance(request, (dict, list)) or hasattr(request, "get"):
        headers = dict(request) if hasattr(request, "items") else request
        if hasattr(headers, "get") and headers.get("Upgrade", "").lower() != "websocket":
            return (
                http.HTTPStatus.OK,
                [("Content-Type", "text/plain"), ("Content-Length", "42")],
                b"OK\nDRISHTI-2.5D Perception Engine Online\n",
            )
        return None

    return None


health_check_handler = process_request
process_http_request = process_request


class TelemetryWebSocketServer:
    """
    Asynchronous WebSocket server broadcasting JSON frames to connected frontend clients.
    Non-blocking: perception loop does not stall even if 0 clients or slow clients are connected.
    Supports concurrent connections and dynamic port routing for cloud deployments (Railway/Render/Docker).
    """

    def __init__(
        self,
        host: str = CONFIG.WS_HOST,
        port: int = CONFIG.WS_PORT,
        ping_interval: float = 30.0,
        ping_timeout: float = 60.0,
        close_timeout: float = 30.0,
    ):
        self.host = host
        self.port = port
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.close_timeout = close_timeout
        self.connected_clients: Set[Any] = set()
        self.server = None
        self._is_running = False
        self.command_callback = None
        self.is_paused = False
        self.playback_speed: float = 1.0
        self.pipeline = None
        self.loader = None

    def set_command_callback(self, callback):
        """Registers a callback for processing frontend HUD playback controls."""
        self.command_callback = callback

    def set_dataset_mode(self, mode: str):
        """Dispatches dataset mode swap to command callback if registered."""
        if self.command_callback is not None:
            cmd = {"action": "set_dataset", "mode": mode}
            try:
                if asyncio.iscoroutinefunction(self.command_callback):
                    asyncio.create_task(self.command_callback(cmd))
                else:
                    self.command_callback(cmd)
            except Exception as e:
                logger.warning(f"[WebSocketServer] Failed to set dataset mode: {e}")
        elif hasattr(self, "pipeline") and self.pipeline is not None and hasattr(self.pipeline, "switch_dataset"):
            self.pipeline.switch_dataset(mode)

    def step_next_frame(self):
        """Advance single frame safely without crashing."""
        if hasattr(self, "pipeline") and self.pipeline is not None:
            if hasattr(self.pipeline, "step_next_frame"):
                self.pipeline.step_next_frame()
            else:
                self.pipeline.is_paused = True
                self.pipeline._step_once = True
        elif hasattr(self, "loader") and self.loader is not None:
            if hasattr(self.loader, "get_next_frame"):
                self.loader.get_next_frame()
            elif hasattr(self.loader, "get_next_sweep"):
                self.loader.get_next_sweep()

    def switch_dataset(self, mode: str):
        """Switches dataset mode safely via pipeline or loader."""
        if hasattr(self, "pipeline") and self.pipeline is not None and hasattr(self.pipeline, "switch_dataset"):
            self.pipeline.switch_dataset(mode)
        elif hasattr(self, "loader") and self.loader is not None and hasattr(self.loader, "switch_dataset"):
            self.loader.switch_dataset(mode)
        elif self.command_callback is not None:
            self.set_dataset_mode(mode)

    def seek_frame(self, target_frame: int):
        """Seeks to target frame index safely."""
        if hasattr(self, "pipeline") and self.pipeline is not None and hasattr(self.pipeline, "seek_frame"):
            self.pipeline.seek_frame(target_frame)
        elif hasattr(self, "loader") and self.loader is not None and hasattr(self.loader, "seek_frame"):
            self.loader.seek_frame(target_frame)

    async def handle_client_message(self, websocket, raw_message):
        """
        Processes client WebSocket commands with full exception guarding
        so NO command can ever crash the connection task or container process.
        """
        try:
            # 1. Parse JSON safely
            if isinstance(raw_message, bytes):
                data = json.loads(raw_message.decode("utf-8"))
            elif isinstance(raw_message, dict):
                data = raw_message
            elif isinstance(raw_message, str):
                try:
                    data = json.loads(raw_message)
                except json.JSONDecodeError:
                    data = {"type": raw_message.strip().lower()}
            else:
                data = {"type": str(raw_message)}
        except Exception as e:
            logger.warning(f"[WS] Malformed incoming message: {e}")
            return

        msg_type = str(data.get("type") or data.get("action") or data.get("command") or "").strip().lower()

        # 2. Safely route commands without throwing unhandled exceptions
        try:
            if msg_type == "ping":
                await websocket.send(json.dumps({"type": "pong"}))

            elif msg_type in ("pause", "playback_pause"):
                self.is_paused = True
                if hasattr(self, "pipeline") and self.pipeline is not None:
                    self.pipeline.is_paused = True
                logger.info("[WS] Telemetry paused by client.")

            elif msg_type in ("resume", "play", "playback_resume", "playback_play"):
                self.is_paused = False
                if hasattr(self, "pipeline") and self.pipeline is not None:
                    self.pipeline.is_paused = False
                logger.info("[WS] Telemetry resumed by client.")

            elif msg_type in ("toggle_pause", "pause_toggle"):
                self.is_paused = not self.is_paused
                if hasattr(self, "pipeline") and self.pipeline is not None:
                    self.pipeline.is_paused = self.is_paused
                logger.info(f"[WS] Telemetry pause toggled to: {self.is_paused}")

            elif msg_type in ("next", "step_forward", "playback_next", "step"):
                # Advance single frame safely
                self.step_next_frame()
                logger.info("[WS] Stepped to next frame.")

            elif msg_type in ("set_dataset", "switch_dataset", "dataset_toggle"):
                mode = str(data.get("mode") or data.get("dataset") or "static").lower()
                logger.info(f"[WS] Switching dataset mode to: {mode}")
                self.switch_dataset(mode)
                sweep_count = 0
                if hasattr(self, "pipeline") and self.pipeline is not None and hasattr(self.pipeline, "dataset_loader"):
                    sweep_count = len(getattr(self.pipeline.dataset_loader, "files", []))
                elif hasattr(self, "loader") and self.loader is not None and hasattr(self.loader, "files"):
                    sweep_count = len(getattr(self.loader, "files", []))
                current_mode = getattr(self.pipeline, "current_dataset_mode", mode) if hasattr(self, "pipeline") and self.pipeline else mode
                try:
                    await websocket.send(json.dumps({
                        "type": "dataset_swapped",
                        "action": "dataset_swapped",
                        "mode": mode,
                        "current_mode": current_mode,
                        "sweep_count": sweep_count
                    }))
                except Exception as ack_err:
                    logger.debug(f"[WS] Acknowledgment send failed: {ack_err}")

            elif msg_type == "seek":
                target_frame = data.get("frame", 0)
                self.seek_frame(target_frame)
                logger.info(f"[WS] Seek to frame: {target_frame}")

            elif msg_type in ("set_speed", "playback_speed", "speed"):
                speed = float(data.get("speed") or data.get("value", 1.0))
                self.playback_speed = max(0.1, min(speed, 5.0))
                if hasattr(self, "pipeline") and self.pipeline is not None:
                    self.pipeline.playback_speed = self.playback_speed
                logger.info(f"[WS] Playback speed set to {self.playback_speed}x")

            # Also invoke registered command_callback if any (e.g. speed adjustment or state sync)
            if self.command_callback is not None:
                try:
                    if asyncio.iscoroutinefunction(self.command_callback):
                        await self.command_callback(data)
                    else:
                        self.command_callback(data)
                except Exception as cb_err:
                    logger.warning(f"[WS Command Error]: {cb_err}")

        except Exception as e:
            logger.error(f"[WS Command Error]: {e}", exc_info=True)
            # Crucial: Send error notification back to client instead of crashing the server
            try:
                await websocket.send(json.dumps({"type": "error", "message": str(e)}))
            except Exception:
                pass

    async def _handler(self, websocket):
        """Registers new client connection and handles incoming messages/heartbeats."""
        self.connected_clients.add(websocket)
        remote = getattr(websocket, "remote_address", "client")
        logger.info(f"[TelemetryServer]: client connected from {remote}. Active clients: {len(self.connected_clients)}")
        try:
            async for raw_message in websocket:
                await self.handle_client_message(websocket, raw_message)
        except (
            websockets.exceptions.ConnectionClosed,
            getattr(websockets.exceptions, "ConnectionClosedOK", ConnectionResetError),
            getattr(websockets.exceptions, "ConnectionClosedError", ConnectionResetError),
            ConnectionResetError,
            asyncio.CancelledError,
        ):
            pass
        except Exception as e:
            if "1006" in str(e) or "abnormal closure" in str(e).lower() or isinstance(e, websockets.exceptions.ConnectionClosedError):
                logger.debug(f"[WebSocketServer] Client {remote} closed unexpectedly: {e}")
            else:
                logger.debug(f"[WebSocketServer] Client {remote} connection closed: {e}")
        finally:
            self.connected_clients.discard(websocket)
            logger.info(f"[TelemetryServer]: client disconnected from {remote}. Active clients: {len(self.connected_clients)}")

    async def broadcast_loop(self, payload_getter=None, target_fps: float = 20.0):
        """
        Safe broadcast loop that continues running and sleeping lightly when paused
        to yield CPU and prevent proxy timeout without pushing frames.
        """
        while self._is_running:
            if self.is_paused:
                await asyncio.sleep(0.05)
                continue

            if payload_getter is not None:
                try:
                    payload = payload_getter()
                    if payload and self.connected_clients:
                        await self.broadcast(payload)
                except Exception as e:
                    logger.error(f"[WS] Error fetching broadcast payload: {e}")

            await asyncio.sleep(1.0 / max(1.0, target_fps))

    _process_request = staticmethod(health_check_handler)
    client_handler = _handler
    handler = _handler

    async def start(self):
        """
        Starts the WebSocket server with automatic ping/pong keep-alive heartbeats
        to prevent cloud edge proxies from closing idle TCP sockets.
        """
        self._loop = asyncio.get_running_loop()
        origins: Optional[list] = None
        if CONFIG.ALLOWED_ORIGINS and CONFIG.ALLOWED_ORIGINS.strip() != "*":
            origins = [o.strip() for o in CONFIG.ALLOWED_ORIGINS.split(",") if o.strip()]

        port = self.port if (self.port is not None and self.port != getattr(CONFIG, "WS_PORT", 8765)) else int(os.environ.get("PORT", getattr(self, "port", 8765)))
        host = getattr(self, "host", "0.0.0.0") or "0.0.0.0"

        self.server = await websockets.serve(
            self.client_handler,
            host=host,
            port=port,
            max_size=15 * 1024 * 1024,      # 15 MB max payload size for dense point/quadtree sweeps
            ping_interval=getattr(self, "ping_interval", 30),
            ping_timeout=getattr(self, "ping_timeout", 60),
            close_timeout=getattr(self, "close_timeout", 30),
            origins=origins,
            process_request=process_request,
        )
        self._is_running = True
        logger.info(f"[WebSocketServer] Listening on ws://{host}:{port} with HTTP probe handling (Max Buffer: 15MB, Ping: 30s, Timeout: 60s)")

    async def broadcast(self, message: Any):
        """
        Broadcasts payload message to all connected clients with non-blocking timeouts
        and automatic cleanup of dead connections.
        """
        if not self.connected_clients:
            return

        if isinstance(message, dict):
            message = fast_serialize(message)
        elif isinstance(message, bytes):
            message = message.decode("utf-8")
        elif not isinstance(message, str):
            message = str(message)

        dead_clients = set()
        # Iterate over a shallow copy to allow safe concurrent modification
        for ws in list(self.connected_clients):
            try:
                # Enforce 150ms timeout per client to prevent queue backup
                await asyncio.wait_for(ws.send(message), timeout=0.15)
            except (websockets.exceptions.ConnectionClosed, asyncio.TimeoutError, Exception) as err:
                logger.debug(f"[WebSocketServer] Dropping unresponsive client {getattr(ws, 'remote_address', 'unknown')}: {err}")
                dead_clients.add(ws)

        if dead_clients:
            self.connected_clients.difference_update(dead_clients)

    def broadcast_nowait(self, message: Any):
        """
        Non-blocking dispatch: schedules the broadcast task concurrently on the running loop
        so the perception compute loop never waits on socket I/O.
        Uses fast binary serialization (orjson) if a dictionary payload is provided.
        Thread-safe across worker threads.
        """
        if self.connected_clients and self._is_running:
            if isinstance(message, dict):
                message = fast_serialize(message)
            elif isinstance(message, bytes):
                message = message.decode("utf-8")

            loop = getattr(self, "_loop", None)
            if loop is not None and not loop.is_closed():
                try:
                    asyncio.run_coroutine_threadsafe(self.broadcast(message), loop)
                    return
                except Exception:
                    pass

            try:
                running_loop = asyncio.get_running_loop()
                running_loop.create_task(self.broadcast(message))
            except Exception:
                pass

    async def stop(self):
        """Gracefully shuts down the server."""
        self._is_running = False
        if self.server:
            self.server.close()
            try:
                await asyncio.wait_for(self.server.wait_closed(), timeout=1.0)
            except (asyncio.TimeoutError, Exception):
                pass
            logger.info("WebSocket server stopped.")


# Alias for backwards compatibility and task conformance
WebSocketServer = TelemetryWebSocketServer
