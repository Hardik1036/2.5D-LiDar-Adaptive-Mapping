"""
Unit tests for Virtual Sensor Replay (backend/ingestion/replay_stream.py)
Verifies that frames loaded from stream are published to registered listeners / queues.
"""

import queue
import pytest
import numpy as np

from backend.ingestion.replay_stream import MockVirtualSensorReplay


def test_mock_virtual_sensor_replay_publishes_to_queue():
    frame_queue = queue.Queue()
    received_frames = []

    def on_frame(points, meta):
        received_frames.append((points, meta))

    mock = MockVirtualSensorReplay(frequency_hz=100.0, on_frame=on_frame, target_queue=frame_queue)
    mock.spin(max_frames=5)

    assert mock.frame_count == 5
    assert len(received_frames) == 5
    assert frame_queue.qsize() == 5

    pts, meta = frame_queue.get_nowait()
    assert isinstance(pts, np.ndarray)
    assert pts.ndim == 2
    assert pts.shape[1] >= 3
