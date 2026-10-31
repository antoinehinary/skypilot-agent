"""Fly a square at constant altitude and save an animated GIF of the flight.

The GIF has two panels: a fixed overview of the square with a trail of markers
along the flown path, and a close chase view of the drone.

Usage:
    python scripts/square_flight.py --gui
    python scripts/square_flight.py --side 1.5 --altitude 1.2 --gif media/square.gif
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from skypilot.drone.backend import MoveResult
from skypilot.drone.sim_backend import SimBackend

logger = logging.getLogger("square_flight")


class FlightRecorder:
    """Step callback that drops trail markers and captures two-panel frames."""

    def __init__(
        self,
        drone: SimBackend,
        overview_target_m: tuple[float, float, float],
        overview_distance_m: float,
        panel_size: tuple[int, int],
        frame_every: int,
        trail_every: int,
    ) -> None:
        self.drone = drone
        self.overview_target_m = overview_target_m
        self.overview_distance_m = overview_distance_m
        self.width, self.height = panel_size
        self.frame_every = frame_every
        self.trail_every = trail_every
        self.frames: list[np.ndarray] = []
        self._step = 0

    def __call__(self) -> None:
        state = self.drone.get_state()
        if state.is_flying and self._step % self.trail_every == 0:
            self.drone.add_marker(state.position_m)
        if self._step % self.frame_every == 0:
            overview = self.drone.render_frame(
                self.width,
                self.height,
                target_m=self.overview_target_m,
                distance_m=self.overview_distance_m,
                pitch_deg=-40.0,
            )
            chase = self.drone.render_frame(self.width, self.height, distance_m=0.6)
            self.frames.append(np.hstack([overview, chase]))
        self._step += 1

    def save(self, path: Path, playback_speed: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame_ms = 1000 * self.frame_every * self.drone.control_period_s / playback_speed
        iio.imwrite(path, self.frames, duration=frame_ms, loop=0)


def fly_square(
    drone: SimBackend, side_m: float, altitude_m: float, speed_mps: float
) -> list[MoveResult]:
    """Take off, fly the square (origin → +x → +x+y → +y → origin), hover, land."""
    results = [drone.takeoff(altitude_m)]
    for x, y in [(side_m, 0.0), (side_m, side_m), (0.0, side_m), (0.0, 0.0)]:
        results.append(drone.goto(x, y, altitude_m, speed_mps=speed_mps))
    results.append(drone.hover(1.0))
    results.append(drone.land())
    return results


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gui", action="store_true", help="show the PyBullet GUI")
    parser.add_argument("--side", type=float, default=1.0, help="square side, m")
    parser.add_argument("--altitude", type=float, default=1.0, help="flight altitude, m")
    parser.add_argument("--speed", type=float, default=0.5, help="horizontal speed, m/s")
    parser.add_argument("--gif", type=Path, default=Path("media/square.gif"), help="output GIF")
    parser.add_argument("--no-gif", action="store_true", help="fly without recording")
    parser.add_argument("--panel-size", type=int, nargs=2, default=(400, 300), metavar=("W", "H"))
    parser.add_argument("--frame-every", type=int, default=4, help="control steps per GIF frame")
    parser.add_argument("--playback-speed", type=float, default=2.0, help="GIF speed-up factor")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with SimBackend(gui=args.gui) as drone:
        recorder = None
        if not args.no_gif:
            half = args.side / 2
            recorder = FlightRecorder(
                drone,
                overview_target_m=(half, half, args.altitude / 2),
                overview_distance_m=0.8 + 1.2 * args.side,
                panel_size=tuple(args.panel_size),
                frame_every=args.frame_every,
                trail_every=2,
            )
            drone.add_step_callback(recorder)

        results = fly_square(drone, args.side, args.altitude, args.speed)
        state = drone.get_state()

    ok = all(r.success for r in results)
    print(
        f"{'Square flight complete' if ok else 'Square flight FAILED'}: "
        f"{state.sim_time_s:.1f} s sim time, battery {state.battery_pct:.1f} %"
    )
    if recorder is not None:
        recorder.save(args.gif, args.playback_speed)
        size_mb = args.gif.stat().st_size / 1e6
        print(f"Saved {args.gif} ({len(recorder.frames)} frames, {size_mb:.1f} MB)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
