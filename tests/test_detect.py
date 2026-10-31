"""Colour detection on synthetic images, plus localisation accuracy in the simulator."""

import math

import cv2
import numpy as np
import pytest

from skypilot.config import load_vision_config
from skypilot.drone.backend import CameraFrame
from skypilot.vision.detect import find_blobs, find_objects
from skypilot.vision.geometry import downward_camera_pose, intrinsics_from_fov, world_to_pixel

CONFIG = load_vision_config()
W, H = 320, 240
FLOOR_WHITE = (224, 224, 224)
FLOOR_BLUE = (160, 185, 224)  # pale blue tile of the rendered floor (saturation ~64)


def _hsv_rgb(h: int, s: int, v: int) -> tuple[int, int, int]:
    pixel = cv2.cvtColor(np.array([[[h, s, v]]], np.uint8), cv2.COLOR_HSV2RGB)
    return tuple(int(c) for c in pixel[0, 0])


def _checker_floor() -> np.ndarray:
    image = np.zeros((H, W, 3), np.uint8)
    image[:] = FLOOR_WHITE
    for r in range(0, H, 40):
        for c in range(0, W, 40):
            if (r // 40 + c // 40) % 2:
                image[r : r + 40, c : c + 40] = FLOOR_BLUE
    return image


def _square(image: np.ndarray, centre: tuple[int, int], side: int, rgb) -> None:
    u, v = centre
    cv2.rectangle(image, (u - side // 2, v - side // 2), (u + side // 2, v + side // 2), rgb, -1)


@pytest.mark.parametrize("colour", sorted(CONFIG.colours))
def test_floor_is_never_detected(colour: str) -> None:
    assert find_blobs(_checker_floor(), CONFIG.colours[colour], CONFIG.min_area_px) == []


@pytest.mark.parametrize("hue", [2, 176])  # red sits on both sides of the hue wrap-around
def test_red_is_detected_across_the_hue_wrap(hue: int) -> None:
    image = _checker_floor()
    _square(image, (100, 60), 20, _hsv_rgb(hue, 220, 180))

    blobs = find_blobs(image, CONFIG.colours["red"], CONFIG.min_area_px)

    assert len(blobs) == 1
    assert blobs[0].centroid_px == pytest.approx((100, 60), abs=0.6)


def test_shaded_faces_of_the_same_colour_are_detected() -> None:
    image = _checker_floor()
    _square(image, (200, 120), 30, _hsv_rgb(63, 208, 96))  # dark green side, measured in sim

    assert len(find_blobs(image, CONFIG.colours["green"], CONFIG.min_area_px)) == 1


def test_small_blobs_are_ignored_and_results_sorted_by_area() -> None:
    image = _checker_floor()
    red = _hsv_rgb(0, 220, 180)
    _square(image, (50, 50), 4, red)  # noise
    _square(image, (150, 100), 12, red)
    _square(image, (250, 180), 24, red)

    blobs = find_blobs(image, CONFIG.colours["red"], CONFIG.min_area_px)

    assert [round(b.centroid_px[0]) for b in blobs] == [250, 150]


def test_find_objects_returns_world_position_and_size() -> None:
    intrinsics = intrinsics_from_fov(W, H, 60.0)
    pose = downward_camera_pose(np.array([1.0, 0.0, 2.0]), yaw_rad=0.0)
    u, v = world_to_pixel(np.array([1.5, -0.4, 0.0]), intrinsics, pose)
    side_px = round(0.3 * intrinsics[0, 0] / 2.0)
    image = _checker_floor()
    _square(image, (round(u), round(v)), side_px, _hsv_rgb(0, 220, 180))

    detections = find_objects(CameraFrame(image, intrinsics, pose, 0.0), "red", CONFIG)

    assert len(detections) == 1
    assert detections[0].position_m == pytest.approx((1.5, -0.4), abs=0.02)
    assert detections[0].size_m == pytest.approx(0.3, abs=0.03)


def test_unknown_colour_raises() -> None:
    frame = CameraFrame(_checker_floor(), intrinsics_from_fov(W, H, 60.0), np.eye(4), 0.0)

    with pytest.raises(KeyError):
        find_objects(frame, "purple", CONFIG)


@pytest.mark.sim
def test_localisation_accuracy_in_the_simulator() -> None:
    pytest.importorskip("pybullet")
    from skypilot.drone.sim_backend import SimBackend

    with SimBackend() as drone:
        red = next(o for o in drone.world.objects if o.colour == "red")
        drone.takeoff(2.5)
        drone.goto(1.0, 0.0, 2.5)  # ~1.1 m away horizontally
        far = find_objects(drone.get_camera_image(), "red", CONFIG)
        drone.goto(*red.position_m, 2.5)  # directly above
        near = find_objects(drone.get_camera_image(), "red", CONFIG)

    assert len(far) == 1 and len(near) == 1
    assert math.dist(far[0].position_m, red.position_m) < 0.15
    assert math.dist(near[0].position_m, red.position_m) < 0.02
