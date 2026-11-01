"""Where to look next: ground coverage of camera frames and unsearched search positions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from skypilot.drone.backend import CameraFrame
from skypilot.vision.geometry import pixel_to_ground


@dataclass(frozen=True, slots=True)
class Rect:
    """Axis-aligned ground rectangle, metres, world frame."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    @property
    def size(self) -> tuple[float, float]:
        return self.x_max - self.x_min, self.y_max - self.y_min


def frame_footprint(frame: CameraFrame) -> Rect | None:
    """Bounding box of the ground area seen in `frame` (None if the corners miss the ground)."""
    height, width = frame.image.shape[:2]
    corners = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]
    points = [pixel_to_ground(u, v, frame.intrinsics, frame.cam_to_world) for u, v in corners]
    if any(p is None for p in points):
        return None
    xs, ys = zip(*points, strict=True)
    return Rect(min(xs), max(xs), min(ys), max(ys))


def search_positions(
    seen: list[Rect],
    area: Rect,
    footprint_size: tuple[float, float],
    current_xy: tuple[float, float],
    count: int = 2,
    spacing_m: float = 0.5,
) -> list[tuple[float, float]]:
    """Up to `count` positions to fly to so the camera sees ground not yet seen.

    `area` is sampled every `spacing_m`; a sample is unseen if it lies in none of the `seen`
    footprints. Greedy cover: repeatedly take the unseen sample nearest to the previous
    position, centre a view of `footprint_size` on it, and drop the samples that view covers.
    An empty list means the whole area has been seen.
    """
    xs = np.arange(area.x_min, area.x_max + 1e-9, spacing_m)
    ys = np.arange(area.y_min, area.y_max + 1e-9, spacing_m)
    unseen = [
        (float(x), float(y)) for x in xs for y in ys if not any(r.contains(x, y) for r in seen)
    ]
    half_x, half_y = footprint_size[0] / 2, footprint_size[1] / 2
    positions: list[tuple[float, float]] = []
    here = current_xy
    while unseen and len(positions) < count:
        cx, cy = min(unseen, key=lambda p: math.dist(p, here))
        positions.append((round(cx, 2), round(cy, 2)))
        unseen = [(x, y) for x, y in unseen if abs(x - cx) > half_x or abs(y - cy) > half_y]
        here = (cx, cy)
    return positions
