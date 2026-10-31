"""End-to-end check of the square-flight demo script (headless)."""

import importlib.util
from pathlib import Path

import imageio.v3 as iio
import pytest

pytest.importorskip("pybullet")

pytestmark = pytest.mark.sim

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "square_flight.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("square_flight", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_square_flight_succeeds_and_writes_gif(tmp_path: Path) -> None:
    gif = tmp_path / "square.gif"

    exit_code = _load_script().main(
        ["--gif", str(gif), "--panel-size", "32", "24", "--frame-every", "24"]
    )

    assert exit_code == 0
    frames = iio.imread(gif, index=None)
    assert frames.shape[1:3] == (24, 64)  # two 32x24 panels side by side
    assert len(frames) > 10
