"""
High-throughput asynchronous WebSocket server for LiDAR perception telemetry.
Cloud-ready with dynamic port binding, concurrent client broadcasting,
and automatic ping-pong heartbeats to keep proxy sockets alive 24/7.
"""

import asyncio
import logging
from typing import Any, Optional, Set
import websockets
import websockets.exceptions

from backend.config import CONFIG, SERVER

logger = logging.getLogger("TelemetryServer")


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

    async def _handler(self, websocket):
        """Registers new client connection and handles incoming messages/heartbeats."""
        self.connected_clients.add(websocket)
        remote = getattr(websocket, "remote_address", "client")
        logger.info(f"Client connected from {remote}. Active concurrent clients: {len(self.connected_clients)}")
        try:
            async for msg in websocket:
                # Can process frontend commands / parameter adjustments if needed
                pass
        except (websockets.exceptions.ConnectionClosed, ConnectionResetError, asyncio.CancelledError):
            pass
        except Exception as e:
            logger.debug(f"Client {remote} connection closed: {e}")
        finally:
            self.connected_clients.discard(websocket)
            logger.info(f"Client disconnected from {remote}. Remaining clients: {len(self.connected_clients)}")

    async def start(self):
        """
        Starts the WebSocket server with automatic ping/pong keep-alive heartbeats
        to prevent cloud edge proxies from closing idle TCP sockets.
        """
        origins: Optional[list] = None
        if CONFIG.ALLOWED_ORIGINS and CONFIG.ALLOWED_ORIGINS.strip() != "*":
            origins = [o.strip() for o in CONFIG.ALLOWED_ORIGINS.split(",") if o.strip()]

        self.server = await websockets.serve(
            self._handler,
            self.host,
            self.port,
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_timeout,
            origins=origins,
        )
        self._is_running = True
        logger.info(
            f"WebSocket telemetry server running at ws://{self.host}:{self.port} "
            f"(ping_interval={self.ping_interval}s, ping_timeout={self.ping_timeout}s)"
        )

    async def broadcast(self, message: str):
        """
        Broadcasts message to all connected clients concurrently.
        Gracefully handles abrupt client disconnections without crashing the perception loop.
        """
        if not self.connected_clients:
            return

        # Prune dead sockets before broadcast
        closed_clients = {c for c in self.connected_clients if getattr(c, "closed", False)}
        if closed_clients:
            self.connected_clients.difference_update(closed_clients)

        if not self.connected_clients:
            return

        try:
            websockets.broadcast(self.connected_clients, message)
        except Exception:
            # Fallback to per-client send with individual error isolation
            disconnected = set()
            for client in list(self.connected_clients):
                try:
                    res = client.send(message)
                    if asyncio.iscoroutine(res):
                        await res
                except (websockets.exceptions.ConnectionClosed, ConnectionResetError, BrokenPipeError):
                    disconnected.add(client)
                except Exception as e:
                    logger.debug(f"Error sending frame to client: {e}")
                    disconnected.add(client)

            if disconnected:
                self.connected_clients.difference_update(disconnected)

    def broadcast_nowait(self, message: str):
        """
        Non-blocking dispatch: schedules the broadcast task concurrently on the running loop
        so the perception compute loop never waits on socket I/O.
        """
        if self.connected_clients and self._is_running:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.broadcast(message))
            except RuntimeError:
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
