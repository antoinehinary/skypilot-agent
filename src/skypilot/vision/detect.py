"""Coloured-object detection: HSV thresholding, blob extraction and ground localisation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from skypilot.config import HsvRange, VisionConfig
from skypilot.drone.backend import CameraFrame
from skypilot.vision.geometry import pixel_to_ground

_OPEN_KERNEL = np.ones((3, 3), np.uint8)  # removes isolated noisy pixels


@dataclass(frozen=True, slots=True)
class PixelBlob:
    """A connected region of matching colour in the image."""

    centroid_px: tuple[float, float]  # (u, v)
    area_px: float


@dataclass(frozen=True, slots=True)
class Detection:
    """A coloured object located on the ground.

    Attributes:
        colour: Colour name, as in config/vision.yaml.
        position_m: Estimated world (x, y) of the object, metres.
        size_m: Approximate width of the object seen from above, metres.
        centroid_px: Blob centre in the image, (u, v).
        area_px: Blob area, pixels.
    """

    colour: str
    position_m: tuple[float, float]
    size_m: float
    centroid_px: tuple[float, float]
    area_px: float


def colour_mask(image_rgb: np.ndarray, ranges: list[HsvRange]) -> np.ndarray:
    """Binary mask (uint8, 0/255) of the pixels inside any of the HSV `ranges`."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for r in ranges:
        mask |= cv2.inRange(hsv, np.array(r.low, np.uint8), np.array(r.high, np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, _OPEN_KERNEL)


def find_blobs(
    image_rgb: np.ndarray, ranges: list[HsvRange], min_area_px: float
) -> list[PixelBlob]:
    """Connected regions matching `ranges`, largest first, ignoring those below `min_area_px`."""
    contours, _ = cv2.findContours(
        colour_mask(image_rgb, ranges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    blobs = []
    for contour in contours:
        moments = cv2.moments(contour)
        area = moments["m00"]
        if area >= min_area_px:
            blobs.append(PixelBlob((moments["m10"] / area, moments["m01"] / area), area))
    return sorted(blobs, key=lambda b: b.area_px, reverse=True)


def find_objects(frame: CameraFrame, colour: str, config: VisionConfig) -> list[Detection]:
    """Detect objects of `colour` in `frame` and locate them on the ground plane (z = 0).

    Raises:
        KeyError: if `colour` has no thresholds in the config.
    """
    # NOTE: a blob touching the image border is only part of the object, so its centroid and size
    # are biased (a 0.3 m box at the frame edge was seen as 0.17 m wide). Flying above it and
    # looking again, as the tool result suggests, removes the bias (README: known limitations).
    blobs = find_blobs(frame.image, config.colours[colour], config.min_area_px)
    camera_height_m = float(frame.cam_to_world[2, 3])
    focal_px = float(frame.intrinsics[0, 0])
    detections = []
    for blob in blobs:
        ground = pixel_to_ground(*blob.centroid_px, frame.intrinsics, frame.cam_to_world)
        if ground is None:
            continue
        size_m = math.sqrt(blob.area_px) * camera_height_m / focal_px
        detections.append(Detection(colour, ground, size_m, blob.centroid_px, blob.area_px))
    return detections
