"""Record a simulated flight as a two-panel GIF: fixed overview with a trail, and a chase view."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from skypilot.drone.backend import Vec3
from skypilot.drone.sim_backend import SimBackend

# Dark grey trail: its HSV value (~40) is below the colour detector's floor (60), so the trail
# can never be mistaken for a coloured object by the onboard camera.
TRAIL_RGBA = (0.15, 0.15, 0.15, 1.0)


class FlightRecorder:
    """Step callback that drops trail markers and captures two-panel frames.

    Attach with `drone.add_step_callback(recorder)`; call `save` at the end.

    Args:
        drone: The simulator to record.
        overview_target_m: Point the fixed overview camera looks at.
        overview_distance_m: Overview camera distance from that point.
        overview_pitch_deg: Overview camera pitch (negative looks down).
        panel_size: (width, height) of each panel, pixels.
        frame_every: Control steps between captured frames.
        trail_every: Control steps between trail markers.
        trail_rgba: Trail marker colour.
        frames_dir: If set, every frame is also written there as a PNG the moment it is
            captured, so a recording survives the process being stopped (see `assemble_gif`).
    """

    def __init__(
        self,
        drone: SimBackend,
        overview_target_m: Vec3,
        overview_distance_m: float,
        overview_pitch_deg: float = -40.0,
        panel_size: tuple[int, int] = (400, 300),
        frame_every: int = 4,
        trail_every: int = 2,
        trail_rgba: tuple[float, float, float, float] = TRAIL_RGBA,
        frames_dir: Path | None = None,
    ) -> None:
        self.drone = drone
        self.overview_target_m = overview_target_m
        self.overview_distance_m = overview_distance_m
        self.overview_pitch_deg = overview_pitch_deg
        self.width, self.height = panel_size
        self.frame_every = frame_every
        self.trail_every = trail_every
        self.trail_rgba = trail_rgba
        self.frames: list[np.ndarray] = []  # kept in memory only without `frames_dir`
        self.frame_count = 0
        self._step = 0
        self.frames_dir = frames_dir
        if frames_dir is not None:
            frames_dir.mkdir(parents=True, exist_ok=True)
            meta = {"frame_s": frame_every * drone.control_period_s}
            (frames_dir / "meta.json").write_text(json.dumps(meta))

    def __call__(self) -> None:
        """Called after every control step: drop a trail marker, capture a frame."""
        state = self.drone.get_state()
        if state.is_flying and self._step % self.trail_every == 0:
            self.drone.add_marker(state.position_m, rgba=self.trail_rgba)
        if self._step % self.frame_every == 0:
            overview = self.drone.render_frame(
                self.width,
                self.height,
                target_m=self.overview_target_m,
                distance_m=self.overview_distance_m,
                pitch_deg=self.overview_pitch_deg,
            )
            chase = self.drone.render_frame(self.width, self.height, distance_m=0.6)
            frame = np.hstack([overview, chase])
            self.frame_count += 1
            if self.frames_dir is None:
                self.frames.append(frame)
            else:  # streamed to disk only: a long mission would not fit in memory
                iio.imwrite(self.frames_dir / f"frame_{self.frame_count:05d}.png", frame)
        self._step += 1

    def save(self, path: Path, playback_speed: float = 2.0) -> None:
        """Write the captured frames as a looping GIF played `playback_speed` times faster."""
        path.parent.mkdir(parents=True, exist_ok=True)
        frame_ms = 1000 * self.frame_every * self.drone.control_period_s / playback_speed
        iio.imwrite(path, self.frames, duration=frame_ms, loop=0)


def assemble_gif(frames_dir: Path, path: Path, playback_speed: float = 2.0) -> int:
    """Build a looping GIF from the PNG frames a `FlightRecorder` wrote to `frames_dir`.

    Returns the number of frames.
    """
    frame_s = json.loads((frames_dir / "meta.json").read_text())["frame_s"]
    files = sorted(frames_dir.glob("frame_*.png"))
    if not files:
        raise FileNotFoundError(f"no frames in {frames_dir}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [iio.imread(f) for f in files]
    iio.imwrite(path, frames, duration=1000 * frame_s / playback_speed, loop=0)
    return len(frames)
