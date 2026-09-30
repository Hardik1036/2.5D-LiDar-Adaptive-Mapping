"""
Unit and Integration Tests for WebSocket Command Dispatcher and Exception Guarding.
Verifies that:
  - All control commands ('pause', 'resume', 'next', 'set_dataset', 'seek', 'ping') execute cleanly.
  - Malformed JSON does not crash the server or connection handler.
  - Unhandled exceptions inside command execution are caught, logged, and return error messages.
  - Telemetry pause, step, and dataset hot-swapping states propagate between WebSocket server and Pipeline.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.server.websocket_server import TelemetryWebSocketServer
from backend.main import PerceptionPipeline


class MockWebSocket:
    def __init__(self):
        self.sent_messages = []

    async def send(self, msg: str):
        self.sent_messages.append(msg)


def test_websocket_server_ping_pong():
    async def _run():
        server = TelemetryWebSocketServer()
        ws = MockWebSocket()

        await server.handle_client_message(ws, json.dumps({"type": "ping"}))
        assert len(ws.sent_messages) == 1
        resp = json.loads(ws.sent_messages[0])
        assert resp.get("type") == "pong"
    asyncio.run(_run())


def test_websocket_server_pause_resume():
    async def _run():
        server = TelemetryWebSocketServer()
        pipeline = PerceptionPipeline()
        server.pipeline = pipeline
        ws = MockWebSocket()

        assert not server.is_paused
        assert not pipeline.is_paused

        # Send pause command
        await server.handle_client_message(ws, json.dumps({"type": "pause"}))
        assert bool(getattr(server, "is_paused"))
        assert bool(getattr(pipeline, "is_paused"))

        # Send resume command
        await server.handle_client_message(ws, json.dumps({"type": "resume"}))
        assert not getattr(server, "is_paused")
        assert not getattr(pipeline, "is_paused")
    asyncio.run(_run())


def test_websocket_server_step_next_frame():
    async def _run():
        server = TelemetryWebSocketServer()
        pipeline = PerceptionPipeline()
        server.pipeline = pipeline
        ws = MockWebSocket()

        # Step frame
        await server.handle_client_message(ws, json.dumps({"type": "next"}))
        assert pipeline.is_paused is True
        assert pipeline._step_once is True
    asyncio.run(_run())


def test_websocket_server_dataset_switch():
    async def _run():
        server = TelemetryWebSocketServer()
        pipeline = PerceptionPipeline()
        server.pipeline = pipeline
        ws = MockWebSocket()

        # Switch to dynamic using "action"
        await server.handle_client_message(ws, json.dumps({"action": "set_dataset", "mode": "dynamic"}))
        assert pipeline.current_dataset_mode == "dynamic"
        assert len(pipeline.dataset_loader.files) > 0
        assert len(ws.sent_messages) == 1
        ack1 = json.loads(ws.sent_messages[0])
        assert ack1.get("type") == "dataset_swapped"
        assert ack1.get("mode") == "dynamic"
        assert ack1.get("sweep_count") > 0

        # Switch back to static using "type"
        await server.handle_client_message(ws, json.dumps({"type": "switch_dataset", "mode": "static"}))
        assert pipeline.current_dataset_mode == "static"
        assert len(pipeline.dataset_loader.files) > 0
        assert len(ws.sent_messages) == 2
        ack2 = json.loads(ws.sent_messages[1])
        assert ack2.get("type") == "dataset_swapped"
        assert ack2.get("mode") == "static"
        assert ack2.get("sweep_count") > 0
    asyncio.run(_run())


def test_websocket_server_seek():
    async def _run():
        server = TelemetryWebSocketServer()
        pipeline = PerceptionPipeline()
        server.pipeline = pipeline
        ws = MockWebSocket()

        target = 10
        await server.handle_client_message(ws, json.dumps({"type": "seek", "frame": target}))
        assert pipeline.dataset_loader.current_index == min(target, len(pipeline.dataset_loader.files) - 1)
    asyncio.run(_run())


def test_websocket_server_set_speed():
    async def _run():
        server = TelemetryWebSocketServer()
        pipeline = PerceptionPipeline()
        server.pipeline = pipeline
        ws = MockWebSocket()

        await server.handle_client_message(ws, json.dumps({"type": "set_speed", "speed": 0.5}))
        assert server.playback_speed == 0.5
        assert pipeline.playback_speed == 0.5

        await server.handle_client_message(ws, json.dumps({"type": "set_speed", "speed": 2.0}))
        assert server.playback_speed == 2.0
        assert pipeline.playback_speed == 2.0
    asyncio.run(_run())


def test_websocket_server_malformed_json_guard():
    async def _run():
        server = TelemetryWebSocketServer()
        ws = MockWebSocket()

        # Test corrupted bytes and strings
        await server.handle_client_message(ws, b"NOT_VALID_JSON{:::}")
        await server.handle_client_message(ws, "{broken json: ...")
        # Must not throw unhandled exception or crash
    asyncio.run(_run())


def test_websocket_server_exception_guard():
    async def _run():
        server = TelemetryWebSocketServer()
        ws = MockWebSocket()

        # Create a failing callback to simulate unexpected runtime errors
        def failing_action(mode: str) -> None:
            raise RuntimeError("Simulated internal perception failure")

        server.switch_dataset = failing_action

        # Should catch error, log, and send error json to websocket without terminating
        await server.handle_client_message(ws, json.dumps({"type": "set_dataset", "mode": "invalid"}))
        assert len(ws.sent_messages) == 1
        err_resp = json.loads(ws.sent_messages[0])
        assert err_resp.get("type") == "error"
        assert "Simulated internal perception failure" in err_resp.get("message", "")
    asyncio.run(_run())
