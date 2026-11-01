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
        """True if the point (x, y), in metres, lies inside the rectangle."""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    @property
    def size(self) -> tuple[float, float]:
        """(width along x, depth along y) in metres."""
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

    Coverage: `area` is sampled every `spacing_m`; a sample is unseen if it lies in none of
    the `seen` footprints, and an empty result means the whole area has been seen.

    Positions: `area` is tiled with the fewest views of `footprint_size` that cover it, each
    view kept inside `area` (a view centred on the edge would waste half its ground). Tiles
    that still contain unseen samples are returned nearest first, each one chained from the
    previous so the list reads as a short route.
    """
    xs = np.arange(area.x_min, area.x_max + 1e-9, spacing_m)
    ys = np.arange(area.y_min, area.y_max + 1e-9, spacing_m)
    unseen = [
        (float(x), float(y))
        for x, y in ((x, y) for x in xs for y in ys)
        if not any(r.contains(x, y) for r in seen)
    ]
    half_x, half_y = footprint_size[0] / 2, footprint_size[1] / 2
    tiles = [
        (round(cx, 2), round(cy, 2))
        for cx in _tile_centres(area.x_min, area.x_max, footprint_size[0])
        for cy in _tile_centres(area.y_min, area.y_max, footprint_size[1])
    ]

    def covers(tile: tuple[float, float], point: tuple[float, float]) -> bool:
        return abs(point[0] - tile[0]) <= half_x and abs(point[1] - tile[1]) <= half_y

    positions: list[tuple[float, float]] = []
    here = current_xy
    while unseen and len(positions) < count:
        useful = [t for t in tiles if t not in positions and any(covers(t, p) for p in unseen)]
        if not useful:
            break
        tile = min(useful, key=lambda t: math.dist(t, here))
        positions.append(tile)
        unseen = [p for p in unseen if not covers(tile, p)]
        here = tile
    return positions


def _tile_centres(low: float, high: float, size: float) -> list[float]:
    """Centres of the fewest views of width `size` covering [low, high], all inside it."""
    n = max(1, math.ceil((high - low) / size))
    if n == 1:
        return [(low + high) / 2]
    return [float(c) for c in np.linspace(low + size / 2, high - size / 2, n)]
