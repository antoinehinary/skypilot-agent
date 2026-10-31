"""Unit tests of the pure safety rules."""

import pytest
import yaml
from pydantic import ValidationError

from skypilot.config import CONFIG_DIR, SafetyConfig, load_safety_config
from skypilot.drone.backend import DroneState
from skypilot.safety.rules import (
    ALLOW,
    validate_goto,
    validate_hover,
    validate_land,
    validate_takeoff,
)

LIMITS = load_safety_config()  # x, y in [-3, 3]; z in [0.2, 3]; 1 m/s; 30 % / 20 %; 300 s


def state(position=(0.0, 0.0, 1.0), battery=100.0, flying=True, sim_time=10.0) -> DroneState:
    return DroneState(position, (0.0, 0.0, 0.0), 0.0, battery, flying, sim_time)


ON_PAD = state(position=(0.0, 0.0, 0.01), flying=False, sim_time=0.0)


# ---------------------------------------------------------------------- config


def test_default_limits() -> None:
    assert (LIMITS.altitude_m.low, LIMITS.altitude_m.high) == (0.2, 3.0)
    assert 3.0 in LIMITS.geofence.x_m and 3.01 not in LIMITS.geofence.x_m


def test_empty_interval_is_rejected() -> None:
    raw = yaml.safe_load((CONFIG_DIR / "safety.yaml").read_text())
    raw["altitude_m"] = [3.0, 0.2]

    with pytest.raises(ValidationError, match="empty"):
        SafetyConfig.model_validate(raw)


# --------------------------------------------------------------------- takeoff


def test_takeoff_allowed_within_limits() -> None:
    assert validate_takeoff(ON_PAD, 1.0, LIMITS) == ALLOW


def test_takeoff_to_50_m_is_refused_with_actionable_reason() -> None:
    decision = validate_takeoff(ON_PAD, 50.0, LIMITS)

    assert not decision.allowed and decision.rule == "altitude"
    assert decision.reason == (
        "target z=50.00 m exceeds the 3.00 m altitude limit; choose z between 0.20 and 3.00 m"
    )


@pytest.mark.parametrize(("battery", "allowed"), [(29.9, False), (30.0, True)])
def test_takeoff_battery_threshold(battery: float, allowed: bool) -> None:
    decision = validate_takeoff(state(battery=battery, flying=False), 1.0, LIMITS)

    assert decision.allowed is allowed
    if not allowed:
        assert decision.rule == "takeoff_battery" and "stay on the ground" in decision.reason


# ------------------------------------------------------------------------ goto


@pytest.mark.parametrize(
    "target",
    [(3.0, -3.0, 3.0), (-3.0, 3.0, 0.2), (0.0, 0.0, 1.5)],  # boundaries are inclusive
)
def test_goto_allowed_inside_limits(target) -> None:
    assert validate_goto(state(), target, 0.5, LIMITS) == ALLOW


@pytest.mark.parametrize(
    ("target", "rule", "fragment"),
    [
        ((0.0, 0.0, 50.0), "altitude", "z=50.00 m exceeds the 3.00 m altitude limit"),
        ((0.0, 0.0, 0.1), "altitude", "below the 0.20 m minimum flight altitude"),
        ((3.1, 0.0, 1.0), "geofence", "x=3.10 m is outside the geofence (x from -3.00 to 3.00 m)"),
        (
            (0.0, -6.0, 1.0),
            "geofence",
            "y=-6.00 m is outside the geofence (y from -3.00 to 3.00 m)",
        ),
    ],
)
def test_goto_outside_limits_is_refused(target, rule, fragment) -> None:
    decision = validate_goto(state(), target, 0.5, LIMITS)

    assert not decision.allowed and decision.rule == rule
    assert fragment in decision.reason
    assert "choose" in decision.reason  # always tells the agent what to do instead


def test_goto_requires_flying() -> None:
    decision = validate_goto(state(flying=False), (1.0, 0.0, 1.0), 0.5, LIMITS)

    assert decision.rule == "not_flying" and "call takeoff first" in decision.reason


@pytest.mark.parametrize(("speed", "allowed"), [(None, True), (1.0, True), (1.5, False)])
def test_goto_speed_limit(speed, allowed: bool) -> None:
    decision = validate_goto(state(), (1.0, 0.0, 1.0), speed, LIMITS)

    assert decision.allowed is allowed
    if not allowed:
        assert decision.rule == "speed" and "use at most 1.00 m/s" in decision.reason


@pytest.mark.parametrize(("battery", "allowed"), [(19.9, False), (20.0, True)])
def test_goto_battery_reserve(battery: float, allowed: bool) -> None:
    decision = validate_goto(state(battery=battery), (1.0, 0.0, 1.0), 0.5, LIMITS)

    assert decision.allowed is allowed
    if not allowed:
        assert decision.rule == "battery_reserve" and "land now" in decision.reason


def test_goto_after_mission_time_is_refused() -> None:
    decision = validate_goto(state(sim_time=300.0), (1.0, 0.0, 1.0), 0.5, LIMITS)

    assert decision.rule == "mission_time" and "land now" in decision.reason


def test_first_failing_rule_wins() -> None:
    decision = validate_goto(state(flying=False, battery=5.0), (0.0, 0.0, 50.0), 9.0, LIMITS)

    assert decision.rule == "not_flying"


# ----------------------------------------------------------------------- hover


def test_hover_that_would_exceed_mission_time_is_refused() -> None:
    decision = validate_hover(state(sim_time=290.0), 30.0, LIMITS)

    assert decision.rule == "mission_time"
    assert "10 s left" in decision.reason and "use at most 10 s or land" in decision.reason


def test_hover_within_mission_time_is_allowed() -> None:
    assert validate_hover(state(sim_time=290.0), 10.0, LIMITS) == ALLOW


# ------------------------------------------------------------------------ land


def test_land_is_never_refused() -> None:
    worst = state(position=(9.0, 9.0, 9.0), battery=0.0, sim_time=10_000.0)

    assert validate_land(worst, LIMITS) == ALLOW


def test_refusal_just_over_a_limit_shows_why() -> None:
    # Regression: a climb to z=3.0127 (measured drift) was refused as "z=3.0 m exceeds 3.0 m".
    decision = validate_goto(state(position=(0.0, 0.0, 1.0127)), (0.0, 0.0, 3.0127), 0.5, LIMITS)

    assert "target z=3.01 m exceeds the 3.00 m altitude limit" in decision.reason
