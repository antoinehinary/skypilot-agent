"""Safety rules as pure functions: (current state, request, limits) -> Decision.

Rules never touch the drone, so every limit and boundary can be unit-tested. Each
refusal says what is wrong and what to do instead, because the reader is an LLM agent
that has to re-plan from it. Landing is never refused: a safety layer must not be able
to trap the drone in the air.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from skypilot.config import SafetyConfig
from skypilot.drone.backend import DroneState, Vec3


@dataclass(frozen=True, slots=True)
class Decision:
    """Outcome of a safety check.

    Attributes:
        allowed: True if the command may be executed.
        rule: Name of the rule that refused the command, or "ok".
        reason: Actionable explanation of a refusal; empty when allowed.
    """

    allowed: bool
    rule: str = "ok"
    reason: str = ""


ALLOW = Decision(True)
Check = Callable[[], Decision | None]


def _refuse(rule: str, reason: str) -> Decision:
    return Decision(False, rule, reason)


def _first_refusal(checks: Iterable[Check]) -> Decision:
    """Run checks in order and return the first refusal, or ALLOW."""
    for check in checks:
        decision = check()
        if decision is not None:
            return decision
    return ALLOW


# --------------------------------------------------------------------------- checks


def check_flying(state: DroneState) -> Decision | None:
    if not state.is_flying:
        return _refuse("not_flying", "the drone is not flying; call takeoff first")
    return None


def check_mission_time(
    state: DroneState, limits: SafetyConfig, extra_s: float = 0.0
) -> Decision | None:
    limit = limits.max_mission_time_s
    if state.sim_time_s >= limit:
        return _refuse(
            "mission_time",
            f"the {limit:.0f} s mission time limit is reached (now {state.sim_time_s:.0f} s); "
            "only land is allowed, land now",
        )
    remaining = limit - state.sim_time_s
    if extra_s > remaining:
        return _refuse(
            "mission_time",
            f"{extra_s:.0f} s would exceed the {limit:.0f} s mission time limit "
            f"({remaining:.0f} s left); use at most {remaining:.0f} s or land",
        )
    return None


def check_battery_reserve(state: DroneState, limits: SafetyConfig) -> Decision | None:
    reserve = limits.battery.reserve_pct
    if state.battery_pct < reserve:
        return _refuse(
            "battery_reserve",
            f"battery {state.battery_pct:.0f} % is below the {reserve:.0f} % reserve; "
            "only land is allowed, land now",
        )
    return None


def check_takeoff_battery(state: DroneState, limits: SafetyConfig) -> Decision | None:
    needed = limits.battery.min_takeoff_pct
    if state.battery_pct < needed:
        return _refuse(
            "takeoff_battery",
            f"battery {state.battery_pct:.0f} % is below the {needed:.0f} % needed to take off; "
            "stay on the ground",
        )
    return None


def check_speed(speed_mps: float | None, limits: SafetyConfig) -> Decision | None:
    limit = limits.max_horizontal_speed_mps
    if speed_mps is not None and speed_mps > limit:
        return _refuse(
            "speed",
            f"speed {speed_mps:.2f} m/s exceeds the {limit:.2f} m/s speed limit; "
            f"use at most {limit:.2f} m/s",
        )
    return None


def check_position(target_m: Vec3, limits: SafetyConfig) -> Decision | None:
    """Target inside the geofence and the altitude band.

    The geofence is a box, hence convex: if the start and the target of a straight move are
    inside it, so is the whole path. Checking the target is enough because the current
    position is inside by induction (takeoff starts on the pad and every move was checked).
    """
    x, y, z = target_m
    alt = limits.altitude_m
    if z > alt.high:
        return _refuse(
            "altitude",
            f"target z={z:.2f} m exceeds the {alt.high:.2f} m altitude limit; "
            f"choose z between {alt.low:.2f} and {alt.high:.2f} m",
        )
    if z < alt.low:
        return _refuse(
            "altitude",
            f"target z={z:.2f} m is below the {alt.low:.2f} m minimum flight altitude; "
            f"choose z between {alt.low:.2f} and {alt.high:.2f} m, or call land to touch down",
        )
    for axis, value, interval in (("x", x, limits.geofence.x_m), ("y", y, limits.geofence.y_m)):
        if value not in interval:
            return _refuse(
                "geofence",
                f"target {axis}={value:.2f} m is outside the geofence "
                f"({axis} from {interval.low:.2f} to {interval.high:.2f} m); "
                f"choose {axis} within that range",
            )
    return None


# ---------------------------------------------------------------------- per command


def validate_takeoff(state: DroneState, altitude_m: float, limits: SafetyConfig) -> Decision:
    x, y, _ = state.position_m
    return _first_refusal(
        [
            lambda: check_mission_time(state, limits),
            lambda: check_takeoff_battery(state, limits),
            lambda: check_position((x, y, altitude_m), limits),
        ]
    )


def validate_goto(
    state: DroneState, target_m: Vec3, speed_mps: float | None, limits: SafetyConfig
) -> Decision:
    return _first_refusal(
        [
            lambda: check_flying(state),
            lambda: check_mission_time(state, limits),
            lambda: check_battery_reserve(state, limits),
            lambda: check_speed(speed_mps, limits),
            lambda: check_position(target_m, limits),
        ]
    )


def validate_hover(state: DroneState, seconds: float, limits: SafetyConfig) -> Decision:
    return _first_refusal(
        [
            lambda: check_flying(state),
            lambda: check_mission_time(state, limits, extra_s=seconds),
            lambda: check_battery_reserve(state, limits),
        ]
    )


def validate_land(state: DroneState, limits: SafetyConfig) -> Decision:
    """Landing is always allowed: it is the safe action of last resort."""
    return ALLOW
