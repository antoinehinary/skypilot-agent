"""Headless tests of SimBackend against the real PyBullet physics."""

from collections.abc import Iterator

import numpy as np
import pytest

pytest.importorskip("pybullet")

from skypilot.config import SimConfig, load_sim_config  # noqa: E402
from skypilot.drone.sim_backend import SimBackend  # noqa: E402

pytestmark = pytest.mark.sim

TOL_M = load_sim_config().motion.position_tolerance_m


def _with_motion(**overrides: float) -> SimConfig:
    config = load_sim_config()
    return config.model_copy(update={"motion": config.motion.model_copy(update=overrides)})


@pytest.fixture
def drone() -> Iterator[SimBackend]:
    with SimBackend() as backend:
        yield backend


@pytest.fixture
def flying(drone: SimBackend) -> SimBackend:
    assert drone.takeoff(1.0).success
    return drone


def test_starts_landed_on_the_pad(drone: SimBackend) -> None:
    state = drone.get_state()

    assert not state.is_flying
    assert state.sim_time_s == pytest.approx(0.0)
    assert state.battery_pct == pytest.approx(100.0)
    assert state.position_m == pytest.approx((0.0, 0.0, 0.0), abs=0.02)


def test_takeoff_reaches_altitude(drone: SimBackend) -> None:
    result = drone.takeoff(1.0)

    assert result.success
    assert drone.get_state().is_flying
    assert result.position_m[2] == pytest.approx(1.0, abs=TOL_M)


def test_goto_reaches_target_in_expected_time(flying: SimBackend) -> None:
    result = flying.goto(1.0, 0.5, 1.5, speed_mps=0.5)

    assert result.success
    assert np.linalg.norm(np.subtract(result.position_m, (1.0, 0.5, 1.5))) <= TOL_M
    distance = np.linalg.norm([1.0, 0.5, 0.5])
    assert distance / 0.5 <= result.elapsed_s <= distance / 0.5 + 1.0


def test_move_relative_offsets_from_current_position(flying: SimBackend) -> None:
    result = flying.move_relative(0.5, -0.5, 0.0)

    assert result.success
    assert result.position_m == pytest.approx((0.5, -0.5, 1.0), abs=TOL_M)


def test_hover_holds_position(flying: SimBackend) -> None:
    before = flying.get_state()

    result = flying.hover(3.0)

    assert result.success
    assert result.elapsed_s == pytest.approx(3.0, abs=flying.control_period_s)
    assert result.position_m == pytest.approx(before.position_m, abs=TOL_M)


def test_land_stops_on_the_ground(flying: SimBackend) -> None:
    result = flying.land()

    state = flying.get_state()
    assert result.success
    assert not state.is_flying
    assert state.position_m[2] == pytest.approx(0.0, abs=0.02)
    assert np.linalg.norm(state.velocity_mps) < 0.01


def test_commands_before_takeoff_fail_without_moving(drone: SimBackend) -> None:
    start = drone.get_state()

    for result in (drone.goto(1.0, 0.0, 1.0), drone.hover(1.0), drone.land()):
        assert not result.success

    assert drone.get_state().position_m == start.position_m
    assert drone.get_state().sim_time_s == start.sim_time_s


def test_second_takeoff_fails(flying: SimBackend) -> None:
    result = flying.takeoff(2.0)

    assert not result.success
    assert "already flying" in result.message


def test_non_positive_speed_fails(flying: SimBackend) -> None:
    assert not flying.goto(1.0, 0.0, 1.0, speed_mps=0.0).success


def test_unreachable_tolerance_times_out() -> None:
    config = _with_motion(position_tolerance_m=1e-6, settle_timeout_s=0.5)
    with SimBackend(config) as backend:
        result = backend.takeoff(1.0)

    assert not result.success
    assert "timed out" in result.message
    # Ramp (1 m at 0.5 m/s) + settle timeout, within one control step.
    assert result.elapsed_s == pytest.approx(2.0 + 0.5, abs=0.05)


def test_battery_drains_only_while_flying_and_faster_when_moving(drone: SimBackend) -> None:
    drone.hover(1.0)  # rejected on the ground
    assert drone.get_state().battery_pct == pytest.approx(100.0)

    drone.takeoff(1.0)
    b0 = drone.get_state().battery_pct
    hover = drone.hover(4.0)
    b1 = drone.get_state().battery_pct
    move = drone.goto(2.0, 0.0, 1.0, speed_mps=0.5)
    b2 = drone.get_state().battery_pct

    assert b0 < 100.0
    hover_rate = (b0 - b1) / hover.elapsed_s
    move_rate = (b1 - b2) / move.elapsed_s
    assert 0 < hover_rate < move_rate


def test_step_callback_runs_every_control_step(drone: SimBackend) -> None:
    calls: list[float] = []
    drone.add_step_callback(lambda: calls.append(drone.get_state().sim_time_s))

    drone.takeoff(0.5)

    steps = round(drone.get_state().sim_time_s / drone.control_period_s)
    assert len(calls) == steps


def test_rendered_frame_is_rgb_image(drone: SimBackend) -> None:
    frame = drone.render_frame(width=64, height=48)

    assert frame.shape == (48, 64, 3)
    assert frame.dtype == np.uint8
    assert frame.std() > 0  # not a blank image


def test_marker_is_visible_and_does_not_collide(drone: SimBackend) -> None:
    view = {"width": 64, "height": 48, "target_m": (0.0, 0.0, 0.5), "distance_m": 0.5}
    before = drone.render_frame(**view)

    drone.add_marker((0.0, 0.0, 0.5), radius_m=0.05)
    after = drone.render_frame(**view)
    drone.takeoff(1.0)  # flies straight up through the marker

    assert np.any(before != after)
    assert drone.get_state().position_m == pytest.approx((0.0, 0.0, 1.0), abs=TOL_M)


def test_runs_are_deterministic() -> None:
    def fly() -> tuple:
        with SimBackend() as backend:
            backend.takeoff(1.0)
            backend.goto(1.0, 1.0, 1.0)
            return backend.get_state().position_m

    assert fly() == fly()


def test_close_is_idempotent() -> None:
    backend = SimBackend()
    backend.close()
    backend.close()


def test_world_boxes_are_rendered_at_their_configured_positions(drone: SimBackend) -> None:
    for obj in drone.world.objects:
        x, y = obj.position_m
        frame = drone.render_frame(
            32, 24, target_m=(x, y, obj.size_m), distance_m=0.3, pitch_deg=-89.0
        )
        centre = frame[12, 16].astype(int)
        expected = np.array(drone.world.colours[obj.colour][:3]) * 255
        dominant = int(np.argmax(expected))
        assert int(np.argmax(centre)) == dominant, (obj.name, centre)


def test_world_boxes_are_solid(drone: SimBackend) -> None:
    box = drone.world.objects[0]
    drone.takeoff(1.0)
    drone.goto(box.position_m[0], box.position_m[1], 1.0)

    result = drone.land()

    assert drone.get_state().position_m[2] == pytest.approx(box.size_m, abs=0.03)
    assert result.success, result.message
