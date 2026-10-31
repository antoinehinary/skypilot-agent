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
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from skypilot.config import CONFIG_DIR, VisionConfig, load_safety_config, load_vision_config
from skypilot.drone.backend import DroneBackend, DroneState, MoveResult
from skypilot.mcp_server.vision_tools import register_vision_tools
from skypilot.safety.guard import SafeDrone

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


def format_result(result: MoveResult, state: DroneState) -> str:
    """Tool result for a movement command: status, message, then the resulting state."""
    status = "REFUSED" if result.refused else ("OK" if result.success else "FAILED")
    return f"{status}: {result.message}. Now {format_state(state)}."


def build_server(
    drone: DroneBackend,
    vision: VisionConfig | None = None,
    image_dir: Path = DEFAULT_IMAGE_DIR,
) -> FastMCP:
    """Create the MCP server with one tool per drone action, bound to `drone`.

    `vision` (default: config/vision.yaml) configures the camera tools; captured images
    are saved under `image_dir`.
    """
    mcp = FastMCP(SERVER_NAME, instructions=INSTRUCTIONS)
    register_vision_tools(mcp, drone, vision or load_vision_config(), image_dir)

    def report(result: MoveResult) -> str:
        text = format_result(result, drone.get_state())
        logger.info(text)
        return text

    @mcp.tool()
    def takeoff(
        altitude_m: Annotated[float, Field(gt=0, description="altitude above the ground, metres")],
    ) -> str:
        """Take off vertically from the ground and hover at `altitude_m` metres.

        Fails if the drone is already flying.
        """
        return report(drone.takeoff(altitude_m))

    @mcp.tool()
    def goto(x_m: Metres, y_m: Metres, z_m: Metres) -> str:
        """Fly in a straight line to the absolute position (x_m, y_m, z_m), in metres.

        Coordinates are in the world frame (x forward, y left, z up, origin at the take-off pad),
        not relative to the drone. Fails if the drone is not flying.
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
        """Descend vertically at the current x, y, touch down and stop the motors."""
        return report(drone.land())

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def get_state() -> str:
        """Return the current position (m), velocity (m/s), battery (%) and flying status.

        Does not move the drone.
        """
        return format_state(drone.get_state())

    return mcp


def main(argv: list[str] | None = None) -> None:
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
    args = parser.parse_args(argv)
    # NOTE: stdout carries the MCP protocol; logging goes to stderr.
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(name)s %(message)s")

    from skypilot.drone.sim_backend import SimBackend  # heavy import, only for the real server

    # The simulator is only ever reachable through the safety guard: there is no bypass flag.
    with SimBackend(gui=args.gui) as sim:
        drone = SafeDrone(
            sim,
            load_safety_config(),
            default_speed_mps=sim.config.motion.default_speed_mps,
            log_path=args.safety_log,
        )
        build_server(drone).run(transport="stdio")


if __name__ == "__main__":
    main()
