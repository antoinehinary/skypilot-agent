"""Shared test fixtures."""

from __future__ import annotations

import math

import numpy as np
import pytest

from skypilot.drone.backend import DroneBackend, DroneState, MoveResult, Vec3


class FakeBackend(DroneBackend):
    """In-memory backend that reaches every target instantly.

    Sim time still advances as if the drone flew at the requested speed, so
    timing-dependent logic in upper layers can be tested without PyBullet.
    """

    def __init__(self) -> None:
        self.position: Vec3 = (0.0, 0.0, 0.0)
        self.is_flying = False
        self.sim_time_s = 0.0
        self.closed = False

    def _fly_to(self, target: Vec3, speed_mps: float) -> MoveResult:
        elapsed = math.dist(self.position, target) / speed_mps
        self.position = target
        self.sim_time_s += elapsed
        return MoveResult(True, target, elapsed, f"reached {target} in {elapsed:.1f} s")

    def takeoff(self, altitude_m: float) -> MoveResult:
        self.is_flying = True
        x, y, _ = self.position
        return self._fly_to((x, y, altitude_m), speed_mps=0.5)

    def goto(
        self, x_m: float, y_m: float, z_m: float, speed_mps: float | None = None
    ) -> MoveResult:
        return self._fly_to((x_m, y_m, z_m), speed_mps or 0.5)

    def hover(self, seconds: float) -> MoveResult:
        self.sim_time_s += seconds
        return MoveResult(True, self.position, seconds, f"hovered {seconds:.1f} s")

    def land(self) -> MoveResult:
        x, y, _ = self.position
        result = self._fly_to((x, y, 0.0), speed_mps=0.3)
        self.is_flying = False
        return result

    def get_state(self) -> DroneState:
        return DroneState(
            position_m=self.position,
            velocity_mps=(0.0, 0.0, 0.0),
            yaw_rad=0.0,
            battery_pct=100.0,
            is_flying=self.is_flying,
            sim_time_s=self.sim_time_s,
        )

    def get_camera_image(self) -> np.ndarray:
        return np.zeros((48, 64, 3), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_drone() -> FakeBackend:
    return FakeBackend()
