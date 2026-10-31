"""Camera tools of the drone MCP server: capture, colour-object search, scene questions."""

# NOTE: no `from __future__ import annotations` here: FastMCP evaluates the tool annotations
# to build their schemas, and find_objects' annotation uses a local variable (`colours`).
import logging
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

logger = logging.getLogger(__name__)

REFINE_HINT = (
    "Estimates are most accurate from directly above the object (error grows to ~0.1 m at "
    "1 m horizontal offset): fly above it and call find_objects again to refine."
)


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


def format_detections(colour: str, detections: list[detect.Detection], frame: CameraFrame) -> str:
    """Tool result for find_objects: positions if found, search advice if not."""
    if not detections:
        return (
            f"No {colour} object in view: {coverage(frame)}. Climb to see more ground, "
            "or move to search another area."
        )
    plural = "s" if len(detections) > 1 else ""
    items = "; ".join(
        f"{colour} #{i} at ({d.position_m[0]:.2f}, {d.position_m[1]:.2f}) m, "
        f"about {d.size_m:.2f} m wide"
        for i, d in enumerate(detections, start=1)
    )
    return f"Found {len(detections)} {colour} object{plural}: {items}. {REFINE_HINT}"


def register_vision_tools(
    mcp: FastMCP, drone: DroneBackend, config: VisionConfig, image_dir: Path
) -> None:
    """Add get_camera_image, find_objects and describe_scene to `mcp`."""
    colours = sorted(config.colours)
    saved = 0

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

        Returns each object's estimated world position (x, y) in metres and approximate width.
        If nothing is found, says what area the camera covers. Does not move the drone.
        """
        colour = colour.strip().lower()
        if colour not in config.colours:
            return f"Unknown colour {colour!r}. Available colours: {', '.join(colours)}."
        frame = drone.get_camera_image()
        text = format_detections(colour, detect.find_objects(frame, colour, config), frame)
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
