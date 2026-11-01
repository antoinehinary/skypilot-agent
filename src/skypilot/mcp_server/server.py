"""MCP server exposing a DroneBackend as tools, over stdio.

Run:
    python -m skypilot.mcp_server.server [--gui]

Tool results are short factual strings written for an LLM: an OK/FAILED status,
what happened, then the resulting position and battery. Argument types and
ranges come from the type hints and are validated before the drone is touched.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from skypilot.config import (
    CONFIG_DIR,
    SafetyConfig,
    VisionConfig,
    load_safety_config,
    load_sim_config,
    load_vision_config,
    load_world_config,
)
from skypilot.drone.backend import DroneBackend, DroneState, MoveResult
from skypilot.mcp_server.vision_tools import SearchArea, register_vision_tools
from skypilot.safety.guard import SafeDrone
from skypilot.vision.search import Rect

if TYPE_CHECKING:
    from skypilot.drone.recording import FlightRecorder
    from skypilot.drone.sim_backend import SimBackend

logger = logging.getLogger(__name__)

SERVER_NAME = "skypilot-drone"
DEFAULT_IMAGE_DIR = CONFIG_DIR.parent / "runs" / "images"
DEFAULT_SAFETY_LOG = CONFIG_DIR.parent / "runs" / "safety_log.jsonl"
INSTRUCTIONS = """\
Controls one quadrotor drone with a downward-facing camera. World frame: x forward, y left, z up,
origin at the take-off pad on the ground. All distances in metres, durations in seconds. Every
movement tool blocks until the move has finished and reports OK or FAILED with the final position,
or REFUSED (the drone did not move) when a safety limit would be violated: geofence, altitude,
speed, battery or mission time. Landing is always allowed. Camera tools locate coloured objects
on the ground. Take off before moving, and land at the end of a mission."""

Metres = Annotated[float, Field(description="metres, world frame")]


def _xyz(v: tuple[float, float, float]) -> str:
    return "({:.2f}, {:.2f}, {:.2f})".format(*(round(c, 2) + 0.0 for c in v))


def format_state(state: DroneState) -> str:
    """One-line description of the drone state for the agent."""
    status = "flying" if state.is_flying else "landed"
    return (
        f"{status} at {_xyz(state.position_m)} m, velocity {_xyz(state.velocity_mps)} m/s, "
        f"battery {state.battery_pct:.1f} %, sim time {state.sim_time_s:.1f} s"
    )


_NUM = r"(-?\d+(?:\.\d+)?)"
_VEC = rf"\({_NUM}, {_NUM}, {_NUM}\)"
_STATE_RE = re.compile(
    rf"(flying|landed) at {_VEC} m, velocity {_VEC} m/s, "
    rf"battery {_NUM} %, sim time {_NUM} s"
)


def parse_state(text: str) -> DroneState:
    """Inverse of `format_state` (values rounded as printed). Yaw is not reported: 0.0.

    Raises:
        ValueError: if `text` does not contain a state in the `format_state` format.
    """
    match = _STATE_RE.search(text)
    if match is None:
        raise ValueError(f"no drone state found in {text!r}")
    status, *numbers = match.groups()
    v = [float(n) for n in numbers]
    return DroneState(
        position_m=(v[0], v[1], v[2]),
        velocity_mps=(v[3], v[4], v[5]),
        yaw_rad=0.0,
        battery_pct=v[6],
        is_flying=status == "flying",
        sim_time_s=v[7],
    )


def format_result(result: MoveResult, state: DroneState) -> str:
    """Tool result for a movement command: status, message, then the resulting state."""
    status = "REFUSED" if result.refused else ("OK" if result.success else "FAILED")
    return f"{status}: {result.message}. Now {format_state(state)}."


def build_server(
    drone: DroneBackend,
    vision: VisionConfig | None = None,
    image_dir: Path = DEFAULT_IMAGE_DIR,
    limits: SafetyConfig | None = None,
) -> FastMCP:
    """Create the MCP server with one tool per drone action, bound to `drone`.

    `vision` (default: config/vision.yaml) configures the camera tools; captured images
    are saved under `image_dir`. With `limits`, find_objects suggests unsearched positions
    inside the geofence and below the altitude ceiling.
    """
    mcp = FastMCP(SERVER_NAME, instructions=INSTRUCTIONS)
    search = None
    if limits is not None:
        fence = limits.geofence
        area = Rect(fence.x_m.low, fence.x_m.high, fence.y_m.low, fence.y_m.high)
        search = SearchArea(area, limits.altitude_m.high)
    register_vision_tools(mcp, drone, vision or load_vision_config(), image_dir, search)

    def report(result: MoveResult) -> str:
        text = format_result(result, drone.get_state())
        logger.info(text)
        return text

    @mcp.tool()
    def takeoff(
        altitude_m: Annotated[float, Field(gt=0, description="altitude above the ground, metres")],
    ) -> str:
        """Take off vertically from the ground and hover at `altitude_m` metres.

        Only from the ground: to change altitude while flying, use goto.
        """
        result = drone.takeoff(altitude_m)
        state = drone.get_state()
        if not result.success and not result.refused and state.is_flying:
            x, y, _ = state.setpoint_m or state.position_m
            call = f"goto(x_m={x:.2f}, y_m={y:.2f}, z_m={altitude_m:.2f})"
            hint = f"to change altitude while flying use {call}"
            result = replace(result, message=f"{result.message}; {hint}")
        return report(result)

    @mcp.tool()
    def goto(x_m: Metres, y_m: Metres, z_m: Metres) -> str:
        """Fly in a straight line to the absolute position (x_m, y_m, z_m), in metres.

        Coordinates are in the world frame (x forward, y left, z up, origin at the take-off pad),
        not relative to the drone. Fails if the drone is not flying. Also changes altitude
        (keep x, y, change z). To touch down on the ground or on an object, use land instead.
        """
        return report(drone.goto(x_m, y_m, z_m))

    @mcp.tool()
    def move_relative(dx_m: Metres, dy_m: Metres, dz_m: Metres) -> str:
        """Fly by (dx_m, dy_m, dz_m) metres from the current position, along the world axes.

        Example: dx_m=1, dy_m=0, dz_m=0 moves 1 m forward along x. Fails if not flying.
        """
        return report(drone.move_relative(dx_m, dy_m, dz_m))

    @mcp.tool()
    def hover(
        seconds: Annotated[float, Field(gt=0, description="hover duration, seconds")],
    ) -> str:
        """Hold the current position for `seconds`. Fails if the drone is not flying."""
        return report(drone.hover(seconds))

    @mcp.tool()
    def land() -> str:
        """Descend vertically at the current x, y and touch down on whatever is below.

        Lands on the ground or on top of an object (e.g. a box): to land on an object, fly
        above it first, then call land. Stops the motors.
        """
        return report(drone.land())

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def get_state() -> str:
        """Return the current position (m), velocity (m/s), battery (%) and flying status.

        Does not move the drone.
        """
        return format_state(drone.get_state())

    return mcp


def main(argv: list[str] | None = None) -> None:
    """Start the drone MCP server over stdio, with the simulator behind SafeDrone."""
    parser = argparse.ArgumentParser(description="SkyPilot drone MCP server (stdio).")
    parser.add_argument("--gui", action="store_true", help="show the PyBullet GUI")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--safety-log",
        type=Path,
        default=DEFAULT_SAFETY_LOG,
        help="JSON Lines file where every safety decision is appended",
    )
    parser.add_argument(
        "--world", type=Path, help="scene objects YAML (default: config/world.yaml)"
    )
    parser.add_argument("--initial-battery", type=float, help="starting battery, percent")
    parser.add_argument(
        "--record-frames", type=Path, help="write recording frames (PNG) to this directory"
    )
    args = parser.parse_args(argv)
    # NOTE: stdout carries the MCP protocol; logging goes to stderr.
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(name)s %(message)s")

    from skypilot.drone.sim_backend import SimBackend  # heavy import, only for the real server

    sim_config = load_sim_config()
    if args.initial_battery is not None:
        battery = sim_config.battery.model_copy(update={"initial_pct": args.initial_battery})
        sim_config = sim_config.model_copy(update={"battery": battery})
    world = load_world_config(args.world) if args.world else None

    # The simulator is only ever reachable through the safety guard: there is no bypass flag.
    with SimBackend(sim_config, gui=args.gui, world=world) as sim:
        limits = load_safety_config()
        drone = SafeDrone(
            sim,
            limits,
            default_speed_mps=sim.config.motion.default_speed_mps,
            log_path=args.safety_log,
        )
        if args.record_frames is not None:
            sim.add_step_callback(_flight_recorder(sim, limits, args.record_frames))
        build_server(drone, limits=limits).run(transport="stdio")


def _flight_recorder(sim: SimBackend, limits: SafetyConfig, frames_dir: Path) -> FlightRecorder:
    """Recorder whose fixed overview frames the whole geofence."""
    from skypilot.drone.recording import FlightRecorder  # heavy import, recording only

    fence = limits.geofence
    centre = ((fence.x_m.low + fence.x_m.high) / 2, (fence.y_m.low + fence.y_m.high) / 2, 0.5)
    span = max(fence.x_m.high - fence.x_m.low, fence.y_m.high - fence.y_m.low)
    return FlightRecorder(
        sim,
        overview_target_m=centre,
        overview_distance_m=1.25 * span,
        overview_pitch_deg=-55.0,
        frame_every=6,
        frames_dir=frames_dir,
    )


if __name__ == "__main__":
    main()
