"""Flight recording streamed to disk and assembled into a GIF."""

import json
from pathlib import Path

import imageio.v3 as iio
import pytest

pytest.importorskip("pybullet")

from skypilot.drone.recording import FlightRecorder, assemble_gif  # noqa: E402
from skypilot.drone.sim_backend import SimBackend  # noqa: E402

pytestmark = pytest.mark.sim


def test_frames_stream_to_disk_and_assemble_into_a_gif(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    with SimBackend() as drone:
        recorder = FlightRecorder(
            drone, (0.0, 0.0, 0.5), 3.0, panel_size=(32, 24), frame_every=12, frames_dir=frames_dir
        )
        drone.add_step_callback(recorder)
        drone.takeoff(1.0)

    assert recorder.frames == []  # nothing kept in memory when streaming
    assert recorder.frame_count == len(list(frames_dir.glob("frame_*.png"))) > 3
    assert json.loads((frames_dir / "meta.json").read_text())["frame_s"] == pytest.approx(0.25)

    gif = tmp_path / "flight.gif"
    count = assemble_gif(frames_dir, gif, playback_speed=2.0)

    frames = iio.imread(gif, index=None)
    assert count == len(frames) == recorder.frame_count
    assert frames.shape[1:3] == (24, 64)  # two 32x24 panels


def test_assemble_without_frames_fails_clearly(tmp_path: Path) -> None:
    (tmp_path / "meta.json").write_text(json.dumps({"frame_s": 0.25}))

    with pytest.raises(FileNotFoundError, match="no frames"):
        assemble_gif(tmp_path, tmp_path / "out.gif")
