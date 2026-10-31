"""Abstract drone interface shared by the simulator and (future) hardware backends.

Everything above this layer (safety guard, MCP server, agent) depends only on
`DroneBackend`, never on PyBullet or cflib directly.

Frame and units: world frame with x forward, y left, z up, origin at the take-off
pad. Positions in metres, velocities in m/s, time in seconds, yaw in radians.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import TracebackType
from typing import Self

import numpy as np

type Vec3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class DroneState:
    """Snapshot of the drone state.

    Attributes:
        position_m: (x, y, z) position in the world frame, metres.
        velocity_mps: (vx, vy, vz) linear velocity in the world frame, m/s.
        yaw_rad: Heading around z, radians, 0 = facing +x.
        battery_pct: Remaining battery, 0-100 %.
        is_flying: True between a successful takeoff and a completed landing.
        sim_time_s: Time since the backend started, seconds (wall time on hardware).
    """

    position_m: Vec3
    velocity_mps: Vec3
    yaw_rad: float
    battery_pct: float
    is_flying: bool
    sim_time_s: float


@dataclass(frozen=True, slots=True)
class MoveResult:
    """Outcome of a blocking movement command.

    Attributes:
        success: True if the target was reached within tolerance before the timeout.
        position_m: Final (x, y, z) position, metres.
        elapsed_s: Duration of the command, seconds.
        message: Short human-readable summary, e.g. "reached (1.00, 0.00, 1.00) in 2.4 s".
    """

    success: bool
    position_m: Vec3
    elapsed_s: float
    message: str


class DroneBackend(ABC):
    """Blocking drone interface: each command returns once it has finished or failed.

    Blocking calls keep the contract simple for the layers above: one call is one
    complete action with a factual result, never a half-executed manoeuvre.
    """

    @abstractmethod
    def takeoff(self, altitude_m: float) -> MoveResult:
        """Take off vertically from the current position and hold `altitude_m` above ground."""

    @abstractmethod
    def goto(self, x_m: float, y_m: float, z_m: float, speed_mps: float = 0.5) -> MoveResult:
        """Fly in a straight line to (x_m, y_m, z_m) in the world frame at `speed_mps`."""

    def move_relative(
        self, dx_m: float, dy_m: float, dz_m: float, speed_mps: float = 0.5
    ) -> MoveResult:
        """Fly by (dx_m, dy_m, dz_m) metres from the current position, world-frame axes."""
        x, y, z = self.get_state().position_m
        return self.goto(x + dx_m, y + dy_m, z + dz_m, speed_mps=speed_mps)

    @abstractmethod
    def hover(self, seconds: float) -> MoveResult:
        """Hold the current position for `seconds`."""

    @abstractmethod
    def land(self) -> MoveResult:
        """Descend vertically, touch down and stop the motors."""

    @abstractmethod
    def get_state(self) -> DroneState:
        """Return the current state without moving the drone."""

    @abstractmethod
    def get_camera_image(self) -> np.ndarray:
        """Return the current onboard camera frame as an (H, W, 3) uint8 RGB array."""

    def close(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Release resources (simulator connection, radio link). Safe to call twice."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
