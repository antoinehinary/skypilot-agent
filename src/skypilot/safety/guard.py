"""SafeDrone: a DroneBackend that checks every command against the safety rules.

It wraps any other backend (simulator or hardware) and exposes the same interface, so
nothing above it changes. Refused commands never reach the wrapped drone. Every decision,
accepted or refused, is logged with a timestamp.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from skypilot.config import SafetyConfig, load_safety_config
from skypilot.drone.backend import CameraFrame, DroneBackend, DroneState, MoveResult
from skypilot.safety import rules
from skypilot.safety.rules import Decision

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """One logged safety decision."""

    timestamp: str  # ISO 8601, UTC wall clock
    sim_time_s: float
    command: str
    arguments: dict[str, Any]
    allowed: bool
    rule: str
    reason: str


class SafeDrone(DroneBackend):
    """Safety guard around `drone`.

    Args:
        drone: The backend that actually flies.
        limits: Safety limits; defaults to `config/safety.yaml`.
        default_speed_mps: Speed the wrapped backend uses when none is requested, so that
            default-speed moves are checked against the speed limit too.
        log_path: Optional JSON Lines file where every decision is appended.
    """

    def __init__(
        self,
        drone: DroneBackend,
        limits: SafetyConfig | None = None,
        default_speed_mps: float | None = None,
        log_path: Path | None = None,
    ) -> None:
        self._drone = drone
        self.limits = limits or load_safety_config()
        self._default_speed_mps = default_speed_mps
        self._log_path = log_path
        self.decisions: list[DecisionRecord] = []

    # ------------------------------------------------------------------ commands

    def takeoff(self, altitude_m: float) -> MoveResult:
        state = self._drone.get_state()
        decision = rules.validate_takeoff(state, altitude_m, self.limits)
        if refusal := self._record("takeoff", {"altitude_m": altitude_m}, state, decision):
            return refusal
        return self._drone.takeoff(altitude_m)

    def goto(
        self, x_m: float, y_m: float, z_m: float, speed_mps: float | None = None
    ) -> MoveResult:
        state = self._drone.get_state()
        speed = speed_mps if speed_mps is not None else self._default_speed_mps
        decision = rules.validate_goto(state, (x_m, y_m, z_m), speed, self.limits)
        arguments = {"x_m": x_m, "y_m": y_m, "z_m": z_m, "speed_mps": speed_mps}
        if refusal := self._record("goto", arguments, state, decision):
            return refusal
        return self._drone.goto(x_m, y_m, z_m, speed_mps)

    def hover(self, seconds: float) -> MoveResult:
        state = self._drone.get_state()
        decision = rules.validate_hover(state, seconds, self.limits)
        if refusal := self._record("hover", {"seconds": seconds}, state, decision):
            return refusal
        return self._drone.hover(seconds)

    def land(self) -> MoveResult:
        state = self._drone.get_state()
        self._record("land", {}, state, rules.validate_land(state, self.limits))
        return self._drone.land()

    # move_relative is inherited: it becomes a goto and is checked there.

    def get_state(self) -> DroneState:
        return self._drone.get_state()

    def get_camera_image(self) -> CameraFrame:
        return self._drone.get_camera_image()

    def close(self) -> None:
        self._drone.close()

    # ------------------------------------------------------------------ internals

    def _record(
        self, command: str, arguments: dict[str, Any], state: DroneState, decision: Decision
    ) -> MoveResult | None:
        """Log the decision; return the refusal result if the command is refused."""
        record = DecisionRecord(
            timestamp=datetime.now(UTC).isoformat(timespec="milliseconds"),
            sim_time_s=state.sim_time_s,
            command=command,
            arguments=arguments,
            allowed=decision.allowed,
            rule=decision.rule,
            reason=decision.reason,
        )
        self.decisions.append(record)
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a") as f:
                f.write(json.dumps(asdict(record)) + "\n")
        if decision.allowed:
            logger.info("safety: accepted %s %s", command, arguments)
            return None
        logger.warning(
            "safety: REFUSED %s %s (%s): %s", command, arguments, decision.rule, decision.reason
        )
        return MoveResult(False, state.position_m, 0.0, decision.reason, refused=True)
