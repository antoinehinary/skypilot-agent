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
) -> list[tuple[float, float]]:
    """Up to `count` unsearched positions inside `area`, nearest to `current_xy` first.

    `area` is tiled with cells the size of one camera footprint; a cell is unsearched if its
    centre lies in none of the `seen` footprints. Centres are kept inside `area`.
    """
    size_x, size_y = footprint_size
    nx, ny = (
        max(1, math.ceil((area.x_max - area.x_min) / size_x)),
        max(1, math.ceil((area.y_max - area.y_min) / size_y)),
    )
    xs = (
        np.linspace(area.x_min + size_x / 2, area.x_max - size_x / 2, nx)
        if nx > 1
        else [(area.x_min + area.x_max) / 2]
    )
    ys = (
        np.linspace(area.y_min + size_y / 2, area.y_max - size_y / 2, ny)
        if ny > 1
        else [(area.y_min + area.y_max) / 2]
    )
    candidates = [
        (round(float(x), 2), round(float(y), 2))
        for x in xs
        for y in ys
        if not any(r.contains(x, y) for r in seen)
    ]
    return sorted(candidates, key=lambda c: math.dist(c, current_xy))[:count]
