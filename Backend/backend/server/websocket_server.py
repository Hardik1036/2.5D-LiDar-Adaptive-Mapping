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
        ping_interval: float = CONFIG.PING_INTERVAL,
        ping_timeout: float = CONFIG.PING_TIMEOUT,
    ):
        self.host = host
        self.port = port
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.connected_clients: Set[Any] = set()
        self.server = None
        self._is_running = False
        self.command_callback = None

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

    async def _handler(self, websocket):
        """Registers new client connection and handles incoming messages/heartbeats."""
        self.connected_clients.add(websocket)
        remote = getattr(websocket, "remote_address", "client")
        logger.info(f"[TelemetryServer]: client connected from {remote}. Active clients: {len(self.connected_clients)}")
        try:
            async for raw_message in websocket:
                try:
                    if isinstance(raw_message, bytes):
                        raw_message = raw_message.decode("utf-8")
                    if isinstance(raw_message, str):
                        try:
                            data = json.loads(raw_message)
                        except json.JSONDecodeError:
                            data = {"command": raw_message.strip().lower()}
                    elif isinstance(raw_message, dict):
                        data = raw_message
                    else:
                        data = {"command": str(raw_message)}

                    msg_type = data.get("type") or data.get("action") or data.get("command")
                    if msg_type == "ping" or (isinstance(raw_message, str) and raw_message.strip().lower() == "ping"):
                        # Immediately reply with pong to keep Railway/reverse-proxy TCP connection alive during pause
                        try:
                            await websocket.send('{"type":"pong"}')
                        except Exception:
                            pass
                        continue

                    action = data.get("action") or data.get("command") or data.get("mode")

                    # Acknowledge step / next / playback commands cleanly
                    if action in ("next", "step"):
                        logger.debug("[WebSocketServer] Stepping single frame")
                    elif action in ("play", "resume", "pause"):
                        logger.debug(f"[WebSocketServer] Playback set to {action}")
                    elif "dataset" in data or "set_dataset" in data or action == "set_dataset":
                        target_mode = data.get("dataset") or data.get("mode")
                        if target_mode in ("static", "dynamic"):
                            self.set_dataset_mode(target_mode)
                        continue

                    if self.command_callback is not None:
                        try:
                            if asyncio.iscoroutinefunction(self.command_callback):
                                await self.command_callback(data)
                            else:
                                self.command_callback(data)
                        except Exception as cmd_err:
                            logger.warning(f"[WebSocketServer] Ignored non-fatal command callback error: {cmd_err}")

                except json.JSONDecodeError:
                    pass
                except Exception as cmd_err:
                    logger.warning(f"[WebSocketServer] Ignored non-fatal command error: {cmd_err}")
        except (
            websockets.exceptions.ConnectionClosed,
            getattr(websockets.exceptions, "ConnectionClosedOK", ConnectionResetError),
            getattr(websockets.exceptions, "ConnectionClosedError", ConnectionResetError),
            ConnectionResetError,
            asyncio.CancelledError,
        ):
            pass
        except Exception as e:
            logger.debug(f"[WebSocketServer] Client {remote} connection closed: {e}")
        finally:
            self.connected_clients.discard(websocket)
            logger.info(f"[TelemetryServer]: client disconnected from {remote}. Active clients: {len(self.connected_clients)}")

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
            max_size=10 * 1024 * 1024,      # 10 MB max payload size for dense point/quadtree sweeps
            ping_interval=getattr(self, "ping_interval", 20),
            ping_timeout=getattr(self, "ping_timeout", 20),
            origins=origins,
            process_request=process_request,
        )
        self._is_running = True
        logger.info(f"[WebSocketServer] Listening on ws://{host}:{port} with HTTP probe handling (Max Buffer: 10MB)")

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
