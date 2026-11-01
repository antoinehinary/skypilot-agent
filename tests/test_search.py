"""Ground coverage and unsearched search positions (pure geometry, no simulator)."""

import numpy as np
import pytest

from skypilot.drone.backend import CameraFrame
from skypilot.vision.geometry import downward_camera_pose, ground_footprint_m, intrinsics_from_fov
from skypilot.vision.search import Rect, frame_footprint, search_positions

K = intrinsics_from_fov(320, 240, 60.0)
AREA = Rect(-3.0, 3.0, -3.0, 3.0)


def _frame(x: float, y: float, z: float) -> CameraFrame:
    return CameraFrame(
        np.zeros((240, 320, 3), np.uint8), K, downward_camera_pose(np.array([x, y, z]), 0.0), 0.0
    )


def test_footprint_is_centred_under_the_camera_with_the_expected_size() -> None:
    rect = frame_footprint(_frame(1.0, -0.5, 2.5))
    across, along = ground_footprint_m(K, (320, 240), 2.5)

    assert ((rect.x_min + rect.x_max) / 2, (rect.y_min + rect.y_max) / 2) == pytest.approx(
        (1.0, -0.5)
    )
    # image height runs along x (forward), image width along y (sideways); pixel centres trim 1 px
    assert rect.size == pytest.approx((along, across), rel=0.01)


def test_every_suggested_view_reveals_unseen_ground() -> None:
    seen = [frame_footprint(_frame(0.0, 0.0, 2.5))]
    size = seen[0].size

    positions = search_positions(seen, AREA, size, current_xy=(0.0, 0.0), count=3)

    assert positions and len(positions) <= 3
    for x, y in positions:
        assert AREA.contains(x, y)
        view = frame_footprint(_frame(x, y, 2.5))
        grid = np.linspace(-3.0, 3.0, 25)
        assert any(
            view.contains(gx, gy) and not seen[0].contains(gx, gy) for gx in grid for gy in grid
        )


def test_a_far_corner_object_is_covered_by_some_search_position() -> None:
    # the moved red box at (-1.9, 1.6) is outside the view from the pad at 2.5 m
    seen = [frame_footprint(_frame(0.0, 0.0, 2.5))]
    size = seen[0].size
    assert not seen[0].contains(-1.9, 1.6)

    positions = search_positions(seen, AREA, size, current_xy=(0.0, 0.0), count=20)

    views = [frame_footprint(_frame(x, y, 2.5)) for x, y in positions]
    assert any(v.contains(-1.9, 1.6) for v in views)


def test_everything_seen_leaves_nothing_to_search() -> None:
    assert search_positions([AREA], AREA, (2.0, 2.0), current_xy=(0.0, 0.0)) == []


def test_area_is_not_covered_while_its_corners_are_unseen() -> None:
    # Regression: from 3 m above the pad the view (3.5 x 4.6 m) contains the centres of a
    # footprint-sized tiling of the 6 x 6 m area, which was wrongly reported as fully covered
    # while the green box at (2.0, 1.5) was out of view.
    seen = [frame_footprint(_frame(0.0, 0.0, 2.5)), frame_footprint(_frame(0.0, 0.0, 3.0))]
    assert not any(r.contains(2.0, 1.5) for r in seen)

    positions = search_positions(seen, AREA, seen[1].size, current_xy=(0.0, 0.0), count=10)

    assert positions, "area wrongly reported as fully covered"
    views = [frame_footprint(_frame(x, y, 3.0)) for x, y in positions]
    assert any(v.contains(2.0, 1.5) for v in views)


def test_search_views_stay_inside_the_area() -> None:
    seen = [frame_footprint(_frame(0.0, 0.0, 2.5))]
    size = frame_footprint(_frame(0.0, 0.0, 3.0)).size  # views planned at the 3 m ceiling

    positions = search_positions(seen, AREA, size, current_xy=(0.0, 0.0), count=10)

    for x, y in positions:
        view = frame_footprint(_frame(x, y, 3.0))
        # positions are rounded to the centimetre
        assert AREA.x_min - 0.01 <= view.x_min and view.x_max <= AREA.x_max + 0.01
        assert AREA.y_min - 0.01 <= view.y_min and view.y_max <= AREA.y_max + 0.01


def test_whole_area_takes_few_views_at_the_ceiling() -> None:
    # 6 x 6 m area, 3.5 x 4.6 m view at 3 m: covering it should take about four views
    seen = [frame_footprint(_frame(0.0, 0.0, 2.5))]
    size = frame_footprint(_frame(0.0, 0.0, 3.0)).size

    positions = search_positions(seen, AREA, size, current_xy=(0.0, 0.0), count=20)
    seen += [frame_footprint(_frame(x, y, 3.0)) for x, y in positions]

    assert len(positions) <= 5
    assert search_positions(seen, AREA, size, current_xy=(0.0, 0.0)) == []
