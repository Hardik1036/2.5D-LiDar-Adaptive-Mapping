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

# Lenient HTTP request parsing to support HEAD, GET, and OPTIONS for cloud health check probes (Render, Railway, etc.)
try:
    from websockets import http11
    from websockets.http11 import parse_line, parse_headers, d

    @classmethod
    def _lenient_http11_parse(cls, read_line):
        try:
            request_line = yield from parse_line(read_line)
        except EOFError as exc:
            raise EOFError("connection closed while reading HTTP request line") from exc
        try:
            method, raw_path, protocol = request_line.split(b" ", 2)
        except ValueError:
            raise ValueError(f"invalid HTTP request line: {d(request_line)}") from None
        if protocol != b"HTTP/1.1":
            raise ValueError(f"unsupported protocol; expected HTTP/1.1: {d(request_line)}")
        if method not in (b"GET", b"HEAD", b"OPTIONS"):
            raise ValueError(f"unsupported HTTP method; expected GET or HEAD; got {d(method)}")
        path = raw_path.decode("ascii", "surrogateescape")
        headers = yield from parse_headers(read_line)
        if "Transfer-Encoding" in headers:
            raise NotImplementedError("transfer codings aren't supported")
        if "Content-Length" in headers:
            raise ValueError("unsupported request body")
        req = cls(path, headers)
        req.method = method.decode("ascii")
        return req

    http11.Request.parse = _lenient_http11_parse
except Exception:
    pass

try:
    from websockets.legacy import http as legacy_http
    from websockets.legacy.http import read_line, read_headers, d

    async def _lenient_legacy_read_request(stream):
        try:
            request_line = await read_line(stream)
        except EOFError as exc:
            raise EOFError("connection closed while reading HTTP request line") from exc
        try:
            method, raw_path, version = request_line.split(b" ", 2)
        except ValueError:
            raise ValueError(f"invalid HTTP request line: {d(request_line)}") from None
        if method not in (b"GET", b"HEAD", b"OPTIONS"):
            raise ValueError(f"unsupported HTTP method: {d(method)}")
        if version != b"HTTP/1.1":
            raise ValueError(f"unsupported HTTP version: {d(version)}")
        path = raw_path.decode("ascii", "surrogateescape")
        headers = await read_headers(stream)
        return path, headers

    legacy_http.read_request = _lenient_legacy_read_request
except Exception:
    pass

try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

from backend.config import CONFIG, SERVER

logger = logging.getLogger("TelemetryServer")


def process_http_request(*args, **kwargs):
    """
    Handles Render / load-balancer health checks (HEAD/GET).
    Compatible with:
      - websockets < 13: process_request(path, headers)
      - websockets >= 13: process_request(connection, request)
    """
    # websockets >= 13.0: args is (connection, request)
    if len(args) == 2 and hasattr(args[1], "headers"):
        connection, request = args
        # Allow normal WebSocket upgrade
        if "Upgrade" in request.headers:
            return None
        # Return 200 OK response using modern websockets response object
        if hasattr(connection, "respond"):
            return connection.respond(http.HTTPStatus.OK, "OK\nDRISHTI-2.5D Perception Engine Online\n")
        return (http.HTTPStatus.OK, [("Content-Type", "text/plain"), ("Content-Length", "42")], b"OK\nDRISHTI-2.5D Perception Engine Online\n")

    # websockets < 13.0: args is (path, headers)
    if len(args) == 2:
        path, headers = args
        if isinstance(headers, dict) or hasattr(headers, "get"):
            if "Upgrade" in headers:
                return None
            return (http.HTTPStatus.OK, [("Content-Type", "text/plain"), ("Content-Length", "42")], b"OK\nDRISHTI-2.5D Perception Engine Online\n")

    return None


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

    async def _handler(self, websocket):
        """Registers new client connection and handles incoming messages/heartbeats."""
        self.connected_clients.add(websocket)
        remote = getattr(websocket, "remote_address", "client")
        logger.info(f"Client connected from {remote}. Active concurrent clients: {len(self.connected_clients)}")
        try:
            async for raw_msg in websocket:
                if self.command_callback is not None:
                    try:
                        if isinstance(raw_msg, bytes):
                            raw_msg = raw_msg.decode("utf-8")
                        if isinstance(raw_msg, str):
                            try:
                                msg_data = json.loads(raw_msg)
                            except Exception:
                                msg_data = {"command": raw_msg.strip().lower()}
                        elif isinstance(raw_msg, dict):
                            msg_data = raw_msg
                        else:
                            msg_data = {"command": str(raw_msg)}

                        if asyncio.iscoroutinefunction(self.command_callback):
                            await self.command_callback(msg_data)
                        else:
                            self.command_callback(msg_data)
                    except Exception as e:
                        logger.debug(f"Command processing error from {remote}: {e}")
        except (websockets.exceptions.ConnectionClosed, ConnectionResetError, asyncio.CancelledError):
            pass
        except Exception as e:
            logger.debug(f"Client {remote} connection closed: {e}")
        finally:
            self.connected_clients.discard(websocket)
            logger.info(f"Client disconnected from {remote}. Remaining clients: {len(self.connected_clients)}")

    _process_request = staticmethod(process_http_request)
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
            self.handler,
            host,
            port,
            max_size=10 * 1024 * 1024,      # 10 MB max payload size for dense point/quadtree sweeps
            ping_interval=getattr(self, "ping_interval", 10),               # Keep alive heartbeat every 10s
            ping_timeout=getattr(self, "ping_timeout", 30),                 # Reconnect drop window
            origins=origins,
            process_request=process_http_request,
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
            if HAS_ORJSON:
                message = orjson.dumps(message).decode("utf-8")
            else:
                import json
                message = json.dumps(message)
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
                if HAS_ORJSON:
                    message = orjson.dumps(message).decode("utf-8")
                else:
                    import json
                    message = json.dumps(message)
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
