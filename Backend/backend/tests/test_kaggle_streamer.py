"""
Unit tests for KaggleDatasetStreamer and Kaggle ingestion in DRISHTI-2.5D.
"""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from backend.ingestion.kaggle_streamer import KaggleDatasetStreamer, KaggleStreamer, DEFAULT_DATASET
from backend.main import PerceptionPipeline


def _create_mock_bin_file(path: Path, num_points: int, channels: int = 4) -> np.ndarray:
    """Helper to generate and save synthetic LiDAR binary data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.arange(num_points * channels, dtype=np.float32).reshape(num_points, channels)
    data.tofile(str(path))
    return data


def test_kaggle_auth_failure():
    """Verify that KaggleDatasetStreamer catches authentication failures gracefully."""
    mock_api = MagicMock()
    mock_api.authenticate.side_effect = Exception("401 Unauthorized - Bad credentials")

    with pytest.raises(RuntimeError) as exc_info:
        KaggleDatasetStreamer(
            dataset_slug="mock/dataset",
            cache_dir="data/test_mock_cache",
            api=mock_api,
        )

    assert "Kaggle authentication failed" in str(exc_info.value)
    assert "401 Unauthorized" in str(exc_info.value)


def test_cache_hit_bypasses_download(tmp_path):
    """When .bin files already exist in cache_dir, do not invoke dataset_download_files."""
    sweep_dir = tmp_path / "sweeps"
    sweep_dir.mkdir(parents=True)
    _create_mock_bin_file(sweep_dir / "000000.bin", num_points=10)

    mock_api = MagicMock()
    mock_api.authenticate.return_value = None

    streamer = KaggleDatasetStreamer(
        dataset_slug="mock/dataset",
        cache_dir=str(tmp_path),
        api=mock_api,
    )

    mock_api.dataset_download_files.assert_not_called()
    assert len(streamer.sweep_files) == 1
    assert streamer.sweep_files[0].name == "000000.bin"


def test_cache_miss_triggers_download(tmp_path):
    """When cache is empty, download files and populate sweeps."""
    mock_api = MagicMock()
    mock_api.authenticate.return_value = None

    def fake_download(slug, path, unzip, quiet):
        # Emulate archive extraction into cache_dir
        _create_mock_bin_file(Path(path) / "frame_01.bin", num_points=20)

    mock_api.dataset_download_files.side_effect = fake_download

    streamer = KaggleDatasetStreamer(
        dataset_slug="mock/dataset",
        cache_dir=str(tmp_path),
        api=mock_api,
    )

    mock_api.dataset_download_files.assert_called_once_with(
        "mock/dataset", path=str(tmp_path), unzip=True, quiet=False
    )
    assert len(streamer.sweep_files) == 1
    assert streamer.sweep_files[0].name == "frame_01.bin"


def test_download_alias_fallback(tmp_path):
    """When primary dataset slug fails with 403/404, fallback alias is attempted."""
    mock_api = MagicMock()
    mock_api.authenticate.return_value = None

    def fake_download(slug, path, unzip, quiet):
        if slug == "hardiknopany1036/drishti-2-5d-dataset":
            raise Exception("403 Forbidden")
        elif slug == "hardiknopany1036/drishti-2-5d-data":
            _create_mock_bin_file(Path(path) / "frame_alias.bin", num_points=25)

    mock_api.dataset_download_files.side_effect = fake_download

    streamer = KaggleDatasetStreamer(
        dataset_slug="hardiknopany1036/drishti-2-5d-dataset",
        cache_dir=str(tmp_path),
        api=mock_api,
    )

    assert streamer.dataset_slug == "hardiknopany1036/drishti-2-5d-data"
    assert len(streamer.sweep_files) == 1
    assert streamer.sweep_files[0].name == "frame_alias.bin"


def test_empty_dataset_raises_file_not_found(tmp_path):
    """If download succeeds but no .bin files exist, raise FileNotFoundError."""
    mock_api = MagicMock()
    mock_api.authenticate.return_value = None

    with pytest.raises(FileNotFoundError) as exc_info:
        KaggleDatasetStreamer(
            dataset_slug="mock/dataset",
            cache_dir=str(tmp_path),
            api=mock_api,
        )

    assert "No valid .bin sweeps found" in str(exc_info.value)


def test_normalization_4_channel(tmp_path):
    """Verify that 4-channel .bin sweeps normalize into (N, 4) float32 arrays."""
    bin_file = tmp_path / "000001.bin"
    expected_data = _create_mock_bin_file(bin_file, num_points=100, channels=4)

    mock_api = MagicMock()
    mock_api.authenticate.return_value = None

    streamer = KaggleDatasetStreamer(
        dataset_slug="mock/dataset",
        cache_dir=str(tmp_path),
        api=mock_api,
    )

    gen = streamer.stream(fps=0)
    pts, meta = next(gen)

    assert pts.shape == (100, 4)
    assert pts.dtype == np.float32
    assert np.allclose(pts, expected_data)
    assert meta["frame_id"] == 0
    assert meta["points_count"] == 100
    assert meta["filename"] == "000001.bin"


def test_normalization_5_channel(tmp_path):
    """Verify that 5-channel .bin sweeps normalize into (N, 4) float32 arrays."""
    bin_file = tmp_path / "000002.bin"
    raw_5ch = _create_mock_bin_file(bin_file, num_points=81, channels=5)

    mock_api = MagicMock()
    mock_api.authenticate.return_value = None

    streamer = KaggleDatasetStreamer(
        dataset_slug="mock/dataset",
        cache_dir=str(tmp_path),
        api=mock_api,
    )

    gen = streamer.stream(fps=0)
    pts, meta = next(gen)

    assert pts.shape == (81, 4)
    assert pts.dtype == np.float32
    assert np.allclose(pts, raw_5ch[:, :4])
    assert meta["points_count"] == 81


def test_normalization_3_channel_fallback(tmp_path):
    """Verify fallback for 3-channel sweeps adds default intensity (15.0)."""
    bin_file = tmp_path / "000003.bin"
    raw_3ch = _create_mock_bin_file(bin_file, num_points=41, channels=3)

    mock_api = MagicMock()
    mock_api.authenticate.return_value = None

    streamer = KaggleDatasetStreamer(
        dataset_slug="mock/dataset",
        cache_dir=str(tmp_path),
        api=mock_api,
    )

    gen = streamer.stream(fps=0)
    pts, meta = next(gen)

    assert pts.shape == (41, 4)
    assert pts.dtype == np.float32
    assert np.allclose(pts[:, :3], raw_3ch)
    assert np.allclose(pts[:, 3], 15.0)


def test_loop_wrapping(tmp_path):
    """Verify loop wrapping when frame_idx >= num_frames."""
    _create_mock_bin_file(tmp_path / "frame_00.bin", num_points=10, channels=4)
    _create_mock_bin_file(tmp_path / "frame_01.bin", num_points=10, channels=4)

    mock_api = MagicMock()
    mock_api.authenticate.return_value = None

    streamer = KaggleDatasetStreamer(
        dataset_slug="mock/dataset",
        cache_dir=str(tmp_path),
        loop=True,
        api=mock_api,
    )

    gen = streamer.stream(fps=0)
    frame_ids = []
    for _ in range(5):
        pts, meta = next(gen)
        frame_ids.append(meta["frame_id"])

    # Expect sequence: 0, 1, 0, 1, 0
    assert frame_ids == [0, 1, 0, 1, 0]


def test_no_loop_terminates(tmp_path):
    """Verify that loop=False terminates after all sweeps are yielded."""
    _create_mock_bin_file(tmp_path / "frame_00.bin", num_points=10, channels=4)
    _create_mock_bin_file(tmp_path / "frame_01.bin", num_points=10, channels=4)

    mock_api = MagicMock()
    mock_api.authenticate.return_value = None

    streamer = KaggleDatasetStreamer(
        dataset_slug="mock/dataset",
        cache_dir=str(tmp_path),
        loop=False,
        api=mock_api,
    )

    gen = streamer.stream(fps=0)
    _, meta0 = next(gen)
    _, meta1 = next(gen)
    assert meta0["frame_id"] == 0
    assert meta1["frame_id"] == 1

    with pytest.raises(StopIteration):
        next(gen)


def test_perception_pipeline_kaggle_integration(tmp_path):
    """Verify PerceptionPipeline seamlessly consumes sweeps from Kaggle stream."""
    _create_mock_bin_file(tmp_path / "000000.bin", num_points=50, channels=4)

    mock_api = MagicMock()
    mock_api.authenticate.return_value = None

    streamer = KaggleDatasetStreamer(
        dataset_slug="mock/dataset",
        cache_dir=str(tmp_path),
        loop=True,
        api=mock_api,
    )

    pipeline = PerceptionPipeline(
        port=8771,
        kaggle=True,
        kaggle_streamer=streamer,
        use_redis=False,
    )

    # Process exactly 2 frames from Kaggle stream
    asyncio.run(pipeline.run(max_frames=2, pacing=False))
    assert pipeline.frame_count == 2


def test_kaggle_missing_credentials(monkeypatch):
    """Verify that KaggleDatasetStreamer raises when credentials are completely missing."""
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    with patch("dotenv.load_dotenv", lambda *args, **kwargs: None), patch("pathlib.Path.home") as mock_home:
        mock_home.return_value = Path("/nonexistent/home")
        with pytest.raises(RuntimeError) as exc_info:
            KaggleDatasetStreamer(
                dataset_slug="mock/dataset",
                cache_dir="data/test_mock_cache",
            )
        assert "Missing Kaggle credentials" in str(exc_info.value)


def test_render_health_check_http_response():
    """Verify HTTP GET health-check returns 200 OK on WebSocket port."""
    import urllib.request
    from backend.server.websocket_server import TelemetryWebSocketServer

    async def _run():
        server = TelemetryWebSocketServer(host="127.0.0.1", port=9199)
        await server.start()
        try:
            def _fetch():
                req = urllib.request.Request("http://127.0.0.1:9199/healthz")
                with urllib.request.urlopen(req) as resp:
                    return resp.status, resp.read().decode("utf-8")

            status, body = await asyncio.to_thread(_fetch)
            assert status == 200
            assert "DRISHTI-2.5D Perception Engine Online" in body
        finally:
            await server.stop()

    asyncio.run(_run())


def test_kaggle_streamer_default_and_sync(tmp_path, monkeypatch):
    """Verify KaggleStreamer defaults to hardiknopany/drishti-2-5d-dataset and handles sync."""
    monkeypatch.delenv("KAGGLE_DATASET", raising=False)
    streamer = KaggleStreamer(cache_dir=str(tmp_path))
    assert streamer.dataset_name == "hardiknopany/drishti-2-5d-dataset"

    # Pre-populate a .bin file to test cache hit
    mock_bin = tmp_path / "000000.bin"
    mock_bin.write_bytes(b"\x00" * 16)
    result_path = streamer.sync_dataset()
    assert result_path == str(tmp_path)


def test_kaggle_streamer_sync_download(tmp_path, monkeypatch):
    """Verify KaggleStreamer downloads dataset when cache is empty."""
    monkeypatch.setenv("KAGGLE_USERNAME", "mock_user")
    monkeypatch.setenv("KAGGLE_KEY", "mock_key")

    mock_api = MagicMock()
    with patch("kaggle.api.kaggle_api_extended.KaggleApi", return_value=mock_api):
        streamer = KaggleStreamer(dataset_name="hardiknopany/drishti-2-5d-dataset", cache_dir=str(tmp_path))
        result_path = streamer.sync_dataset()
        assert result_path == str(tmp_path)
        mock_api.authenticate.assert_called_once()
        mock_api.dataset_download_files.assert_called_once_with(
            "hardiknopany/drishti-2-5d-dataset", path=str(tmp_path), unzip=True
        )


