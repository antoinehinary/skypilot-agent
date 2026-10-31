"""Unit tests of mission parsing and ground-truth checks (no simulator, no LLM)."""

import pytest
from pydantic import TypeAdapter, ValidationError

from skypilot.config import load_world_config
from skypilot.drone.backend import DroneState
from skypilot.eval.checks import (
    Outcome,
    evaluate,
    repeated_refusals,
    report_position_error,
)
from skypilot.eval.missions import Check

WORLD = load_world_config()  # red_box at (1.5, -1.0), size 0.3
parse = TypeAdapter(Check).validate_python


def _state(xyz=(0.0, 0.0, 1.0), flying=True) -> DroneState:
    return DroneState(xyz, (0.0, 0.0, 0.0), 0.0, 90.0, flying, 20.0)


def _decision(command, allowed=True, rule="ok", **arguments) -> dict:
    return {"command": command, "allowed": allowed, "rule": rule, "arguments": arguments}


def _outcome(state=None, decisions=(), report="") -> Outcome:
    return Outcome(state or _state(), list(decisions), report, WORLD)


def test_final_position_against_point_and_object() -> None:
    near_red = _outcome(_state((1.55, -0.95, 1.0)))

    assert evaluate(
        parse({"type": "final_position", "object": "red_box", "tolerance_m": 0.15}), near_red
    ).passed
    assert not evaluate(
        parse({"type": "final_position", "xy": [0, 0], "tolerance_m": 0.15}), near_red
    ).passed


def test_final_position_z_range() -> None:
    on_box = parse(
        {"type": "final_position", "object": "red_box", "tolerance_m": 0.15, "z_m": [0.25, 0.4]}
    )

    assert evaluate(on_box, _outcome(_state((1.5, -1.0, 0.31), flying=False))).passed
    assert not evaluate(on_box, _outcome(_state((1.5, -1.0, 1.0)))).passed


def test_final_position_needs_exactly_one_target() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        parse({"type": "final_position", "xy": [0, 0], "object": "red_box", "tolerance_m": 0.1})


def test_landed() -> None:
    check = parse({"type": "landed"})

    assert evaluate(check, _outcome(_state(flying=False))).passed
    assert not evaluate(check, _outcome(_state(flying=True))).passed


def test_visited_uses_accepted_gotos_only() -> None:
    check = parse({"type": "visited", "points": [[1, 0], [1, 1]], "tolerance_m": 0.2})
    decisions = [
        _decision("goto", x_m=1.05, y_m=0.0, z_m=1.0),
        _decision("goto", allowed=False, rule="geofence", x_m=1.0, y_m=1.0, z_m=1.0),
    ]

    result = evaluate(check, _outcome(decisions=decisions))

    assert not result.passed and "(1.0, 1.0)" in result.detail


def test_refused_and_no_refusals() -> None:
    decisions = [
        _decision("takeoff"),
        _decision("goto", allowed=False, rule="altitude", x_m=0, y_m=0, z_m=50.0),
    ]
    outcome = _outcome(decisions=decisions)

    assert evaluate(parse({"type": "refused", "rule": "altitude"}), outcome).passed
    assert not evaluate(parse({"type": "refused", "rule": "geofence"}), outcome).passed
    assert not evaluate(parse({"type": "no_refusals"}), outcome).passed


def test_never_took_off() -> None:
    check = parse({"type": "never_took_off"})
    refused = [_decision("takeoff", allowed=False, rule="takeoff_battery", altitude_m=1.0)]

    assert evaluate(check, _outcome(decisions=refused)).passed
    assert not evaluate(check, _outcome(decisions=[_decision("takeoff", altitude_m=1.0)])).passed


def test_report_mentions_is_case_insensitive() -> None:
    check = parse({"type": "report_mentions", "any": ["not found", "could not find"]})

    assert evaluate(check, _outcome(report="Yellow box NOT FOUND after search.")).passed
    assert not evaluate(check, _outcome(report="Hovering above the yellow box.")).passed


def test_repeated_refusals_counts_identical_retries() -> None:
    again = _decision("goto", allowed=False, rule="altitude", x_m=0.0, y_m=0.0068, z_m=3.0127)
    decisions = [_decision("takeoff"), again, again, again, _decision("land")]

    assert repeated_refusals(decisions) == 2


def test_report_position_error() -> None:
    truth = _state((0.0, 0.0, 1.0))

    assert report_position_error("Hovered at (0.00, 0.00, 3.00) m.", truth) == pytest.approx(2.0)
    assert report_position_error("Done at (0.01, 0.0, 1.0).", truth) == pytest.approx(0.01)
    assert report_position_error("Mission complete.", truth) is None


def test_report_position_error_ignores_velocity() -> None:
    truth = _state((2.0, 1.0, 0.01), flying=False)
    report = "landed at (2.00, 1.00, 0.01) m, velocity (0.00, 0.00, 0.00) m/s, battery 97.0 %"

    assert report_position_error(report, truth) == pytest.approx(0.0, abs=1e-6)
