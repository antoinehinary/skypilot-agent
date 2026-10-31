"""Pinhole camera geometry: build camera models and map between pixels and the world.

Conventions follow `skypilot.drone.backend.CameraFrame`: camera z along the optical
axis, x to image right, y to image bottom; pixel centres at integer (u, v).
"""

from __future__ import annotations

import math

import numpy as np


def intrinsics_from_fov(width: int, height: int, vertical_fov_deg: float) -> np.ndarray:
    """Pinhole matrix K for square pixels and a vertical field of view, in pixels."""
    f = (height / 2) / math.tan(math.radians(vertical_fov_deg) / 2)
    return np.array([[f, 0.0, (width - 1) / 2], [0.0, f, (height - 1) / 2], [0.0, 0.0, 1.0]])


def downward_camera_pose(position_m: np.ndarray, yaw_rad: float) -> np.ndarray:
    """4x4 camera-to-world transform of a level camera looking straight down.

    The image top points along the heading `yaw_rad` (a yaw-only gimbal: the image stays
    level when the drone rolls or pitches).
    """
    forward = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0])
    z_axis = np.array([0.0, 0.0, -1.0])  # optical axis: down
    y_axis = -forward  # image bottom: backwards
    x_axis = np.cross(y_axis, z_axis)  # image right: the drone's right
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    pose[:3, 3] = position_m
    return pose


def world_to_pixel(
    point_m: np.ndarray, intrinsics: np.ndarray, cam_to_world: np.ndarray
) -> tuple[float, float] | None:
    """Project a world point to (u, v) pixel coordinates; None if it is behind the camera."""
    world_to_cam = np.linalg.inv(cam_to_world)
    p_cam = world_to_cam[:3, :3] @ np.asarray(point_m, dtype=float) + world_to_cam[:3, 3]
    if p_cam[2] <= 0:
        return None
    u, v, w = intrinsics @ p_cam
    return float(u / w), float(v / w)


def pixel_to_ground(
    u: float,
    v: float,
    intrinsics: np.ndarray,
    cam_to_world: np.ndarray,
    plane_z_m: float = 0.0,
) -> tuple[float, float] | None:
    """Intersect the viewing ray through pixel (u, v) with the horizontal plane z = `plane_z_m`.

    Returns the world (x, y) of the intersection, or None if the ray does not hit the
    plane in front of the camera (e.g. a pixel above the horizon).
    """
    ray_cam = np.linalg.solve(intrinsics, np.array([u, v, 1.0]))
    ray_world = cam_to_world[:3, :3] @ ray_cam
    origin = cam_to_world[:3, 3]
    if abs(ray_world[2]) < 1e-9:
        return None
    t = (plane_z_m - origin[2]) / ray_world[2]
    if t <= 0:
        return None
    hit = origin + t * ray_world
    return float(hit[0]), float(hit[1])


def ground_footprint_m(
    intrinsics: np.ndarray, image_size: tuple[int, int], height_m: float
) -> tuple[float, float]:
    """Width and depth (metres) of the ground area seen by a downward camera at `height_m`."""
    width, height = image_size
    f = float(intrinsics[0, 0])
    return height_m * width / f, height_m * height / f
