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

from skypilot.drone.backend import MoveResult
from skypilot.drone.recording import FlightRecorder
from skypilot.drone.sim_backend import SimBackend

logger = logging.getLogger("square_flight")


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
                trail_rgba=(1.0, 0.4, 0.0, 1.0),  # orange: no colour detection in this demo
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
