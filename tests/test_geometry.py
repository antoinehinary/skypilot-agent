"""Pinhole geometry on synthetic cameras: no simulator involved."""

import math

import numpy as np
import pytest

from skypilot.vision.geometry import (
    downward_camera_pose,
    ground_footprint_m,
    intrinsics_from_fov,
    pixel_to_ground,
    world_to_pixel,
)

W, H, FOV = 320, 240, 60.0
K = intrinsics_from_fov(W, H, FOV)
CX, CY = (W - 1) / 2, (H - 1) / 2


def test_intrinsics_from_vertical_fov() -> None:
    f = (H / 2) / math.tan(math.radians(FOV / 2))

    np.testing.assert_allclose(K, [[f, 0, CX], [0, f, CY], [0, 0, 1]])


def test_image_centre_looks_straight_down() -> None:
    pose = downward_camera_pose(np.array([1.0, -2.0, 2.5]), yaw_rad=0.3)

    assert pixel_to_ground(CX, CY, K, pose) == pytest.approx((1.0, -2.0))


def test_image_up_is_forward_and_image_right_is_right() -> None:
    pose = downward_camera_pose(np.array([0.0, 0.0, 2.0]), yaw_rad=0.0)

    up_x, up_y = pixel_to_ground(CX, CY - 50, K, pose)
    right_x, right_y = pixel_to_ground(CX + 50, CY, K, pose)

    assert up_x > 0 and up_y == pytest.approx(0.0)  # forward = +x
    assert right_y < 0 and right_x == pytest.approx(0.0)  # right = -y (y is left)


def test_yaw_rotates_the_image_with_the_drone() -> None:
    pose = downward_camera_pose(np.array([0.0, 0.0, 2.0]), yaw_rad=math.pi / 2)

    x, y = pixel_to_ground(CX, CY - 50, K, pose)

    assert y > 0 and x == pytest.approx(0.0, abs=1e-9)  # heading +y: image up = +y


def test_offset_scales_with_height_over_focal_length() -> None:
    pose = downward_camera_pose(np.array([0.0, 0.0, 2.0]), yaw_rad=0.0)

    x, _ = pixel_to_ground(CX, CY - 100, K, pose)

    assert x == pytest.approx(2.0 * 100 / K[1, 1])


def test_round_trip_world_to_pixel_to_ground() -> None:
    rng = np.random.default_rng(0)
    pose = downward_camera_pose(np.array([0.5, 0.2, 1.7]), yaw_rad=0.7)

    for _ in range(200):
        point = np.array([*rng.uniform(-1.0, 1.0, size=2), 0.0])
        u, v = world_to_pixel(point, K, pose)
        assert pixel_to_ground(u, v, K, pose) == pytest.approx(tuple(point[:2]))


def test_projection_onto_a_raised_plane() -> None:
    pose = downward_camera_pose(np.array([0.0, 0.0, 2.0]), yaw_rad=0.0)
    box_top = np.array([0.4, -0.3, 0.3])

    u, v = world_to_pixel(box_top, K, pose)

    assert pixel_to_ground(u, v, K, pose, plane_z_m=0.3) == pytest.approx((0.4, -0.3))
    ground_x, _ = pixel_to_ground(u, v, K, pose)  # wrong plane: overestimates the offset
    assert ground_x == pytest.approx(0.4 * 2.0 / 1.7)


def test_points_behind_the_camera_or_above_the_horizon_are_rejected() -> None:
    pose = downward_camera_pose(np.array([0.0, 0.0, 2.0]), yaw_rad=0.0)

    assert world_to_pixel(np.array([0.0, 0.0, 3.0]), K, pose) is None
    assert pixel_to_ground(CX, CY, K, pose, plane_z_m=5.0) is None


def test_ground_footprint() -> None:
    width_m, depth_m = ground_footprint_m(K, (W, H), height_m=2.5)

    assert depth_m == pytest.approx(2 * 2.5 * math.tan(math.radians(FOV / 2)))
    assert width_m == pytest.approx(depth_m * W / H)
