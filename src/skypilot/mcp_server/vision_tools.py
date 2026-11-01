"""Camera tools of the drone MCP server: capture, colour-object search, scene questions."""

# NOTE: no `from __future__ import annotations` here: FastMCP evaluates the tool annotations
# to build their schemas, and find_objects' annotation uses a local variable (`colours`).
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import cv2
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from skypilot.config import VisionConfig
from skypilot.drone.backend import CameraFrame, DroneBackend
from skypilot.vision import describe, detect
from skypilot.vision.geometry import ground_footprint_m
from skypilot.vision.search import Rect, frame_footprint, search_positions

logger = logging.getLogger(__name__)

ACCURACY_NOTE = (
    "estimates are most accurate from directly above "
    "(about 0.1 m off when seen from 1 m to the side)"
)


@dataclass(frozen=True)
class SearchArea:
    """Where the drone may search: the geofence and the altitude ceiling."""

    area: Rect
    max_altitude_m: float


def coverage(frame: CameraFrame) -> str:
    """Describe the ground area seen by a downward camera frame."""
    height, width = frame.image.shape[:2]
    altitude_m = float(frame.cam_to_world[2, 3])
    across_m, along_m = ground_footprint_m(frame.intrinsics, (width, height), altitude_m)
    x, y = frame.cam_to_world[:2, 3]
    return (
        f"the camera (altitude {altitude_m:.2f} m) sees about {along_m:.1f} m (forward) x "
        f"{across_m:.1f} m (sideways) of ground centred on ({x:.2f}, {y:.2f})"
    )


def format_detections(colour: str, detections: list[detect.Detection], altitude_m: float) -> str:
    """Tool result when objects are found: positions, then the call that flies above #1.

    Like safety refusals, the result ends with a ready-to-use call: small models copy a
    concrete call far more reliably than they work out the next move themselves.
    """
    plural = "s" if len(detections) > 1 else ""
    items = "; ".join(
        f"{colour} #{i} at ({d.position_m[0]:.2f}, {d.position_m[1]:.2f}) m, "
        f"about {d.size_m:.2f} m wide"
        for i, d in enumerate(detections, start=1)
    )
    x, y = detections[0].position_m
    return (
        f"Found {len(detections)} {colour} object{plural}: {items}. Next: fly above #1 with "
        f"goto(x_m={x:.2f}, y_m={y:.2f}, z_m={altitude_m:.2f}), then call find_objects again "
        f"there to refine, since {ACCURACY_NOTE}."
    )


def format_not_found(
    colour: str,
    frame: CameraFrame,
    altitude_m: float,
    search: SearchArea | None,
    seen: list[Rect],
) -> str:
    """Tool result when nothing is found: what was covered and where to look next."""
    text = f"No {colour} object in view: {coverage(frame)}."
    if search is None:
        return f"{text} Climb to see more ground, or move to search another area."
    if altitude_m < search.max_altitude_m - 0.1:
        text += f" Climbing (up to z={search.max_altitude_m:.2f} m) shows more ground."
    current = frame_footprint(frame)
    x, y = frame.cam_to_world[:2, 3]
    positions = search_positions(seen, search.area, current.size, (float(x), float(y)))
    if not positions:
        return f"{text} The whole search area has been covered: there is no {colour} object."
    calls = " or ".join(
        f"goto(x_m={px:.2f}, y_m={py:.2f}, z_m={altitude_m:.2f})" for px, py in positions
    )
    return f"{text} Unsearched areas, nearest first: {calls}, then call find_objects again."


def register_vision_tools(
    mcp: FastMCP,
    drone: DroneBackend,
    config: VisionConfig,
    image_dir: Path,
    search: SearchArea | None = None,
) -> None:
    """Add get_camera_image, find_objects and describe_scene to `mcp`.

    With a `search` area, find_objects remembers the ground it has covered and suggests
    unsearched positions when nothing is found.
    """
    colours = sorted(config.colours)
    saved = 0
    seen: list[Rect] = []

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def get_camera_image() -> str:
        """Capture the downward-facing camera and save it as a PNG.

        Returns the file path and the ground area covered. The top of the image is the drone's
        forward direction (+x when the drone has not turned). Does not move the drone.
        """
        nonlocal saved
        frame = drone.get_camera_image()
        image_dir.mkdir(parents=True, exist_ok=True)
        saved += 1
        path = image_dir / f"frame_{saved:03d}_t{frame.sim_time_s:06.1f}s.png"
        cv2.imwrite(str(path), cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR))
        height, width = frame.image.shape[:2]
        return f"Saved {width}x{height} image to {path}; {coverage(frame)}."

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def find_objects(
        colour: Annotated[str, Field(description=f"object colour, one of: {', '.join(colours)}")],
    ) -> str:
        """Find objects of a given colour in the downward camera view.

        Returns each object's estimated world position (x, y) in metres and approximate width,
        and the goto call that flies above it. If nothing is found, says what area the camera
        covers and suggests unsearched positions. Does not move the drone.
        """
        colour = colour.strip().lower()
        if colour not in config.colours:
            return f"Unknown colour {colour!r}. Available colours: {', '.join(colours)}."
        state = drone.get_state()
        if not state.is_flying:
            return "The drone is on the ground: take off first, e.g. takeoff(altitude_m=2.5)."
        altitude_m = (state.setpoint_m or state.position_m)[2]
        frame = drone.get_camera_image()
        if (footprint := frame_footprint(frame)) is not None:
            seen.append(footprint)
        detections = detect.find_objects(frame, colour, config)
        if detections:
            text = format_detections(colour, detections, altitude_m)
        else:
            text = format_not_found(colour, frame, altitude_m, search, seen)
        logger.info(text)
        return text

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def describe_scene(
        question: Annotated[str, Field(description="question about what the camera sees")],
    ) -> str:
        """Ask a vision-language model a question about the current downward camera view.

        Slow (several seconds) and approximate: use it for qualitative questions only. For object
        positions, use find_objects instead. Does not move the drone.
        """
        try:
            return describe.describe_scene(drone.get_camera_image(), question, config.describe)
        except describe.VisionModelError as e:
            return f"FAILED: {e}"
