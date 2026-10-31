"""Tests of the DroneBackend contract, using the in-memory FakeBackend."""

import dataclasses

import pytest

from skypilot.drone.backend import DroneBackend, DroneState


def test_backend_is_abstract() -> None:
    with pytest.raises(TypeError):
        DroneBackend()  # type: ignore[abstract]


def test_move_relative_is_offset_from_current_position(fake_drone) -> None:
    fake_drone.takeoff(1.0)
    fake_drone.goto(1.0, 2.0, 1.0)

    result = fake_drone.move_relative(0.5, -1.0, 0.5)

    assert result.success
    assert result.position_m == pytest.approx((1.5, 1.0, 1.5))
    assert fake_drone.get_state().position_m == pytest.approx((1.5, 1.0, 1.5))


def test_move_relative_forwards_speed(fake_drone) -> None:
    fake_drone.takeoff(1.0)

    result = fake_drone.move_relative(2.0, 0.0, 0.0, speed_mps=1.0)

    assert result.elapsed_s == pytest.approx(2.0)


def test_state_is_immutable(fake_drone) -> None:
    state = fake_drone.get_state()

    with pytest.raises(dataclasses.FrozenInstanceError):
        state.battery_pct = 0.0  # type: ignore[misc]


def test_context_manager_closes_backend(fake_drone) -> None:
    with fake_drone as drone:
        drone.takeoff(1.0)

    assert fake_drone.closed


def test_move_relative_starts_from_the_setpoint_not_the_measured_position(fake_drone) -> None:
    fake_drone.takeoff(1.0)
    fake_drone.position = (0.004, 0.007, 1.013)  # measured drift; setpoint stays (0, 0, 1)
    fake_drone.get_state = lambda: DroneState(
        (0.004, 0.007, 1.013), (0.0, 0.0, 0.0), 0.0, 100.0, True, 2.0, setpoint_m=(0.0, 0.0, 1.0)
    )

    result = fake_drone.move_relative(0.0, 0.0, 2.0)

    assert result.position_m == pytest.approx((0.0, 0.0, 3.0))
