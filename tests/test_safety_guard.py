"""SafeDrone around FakeBackend: refused commands never reach the drone."""

import asyncio
import json
from pathlib import Path

import pytest

from skypilot.drone.backend import DroneState
from skypilot.mcp_server.server import build_server
from skypilot.safety.guard import SafeDrone


@pytest.fixture
def safe(fake_drone, tmp_path: Path) -> SafeDrone:
    return SafeDrone(fake_drone, default_speed_mps=0.5, log_path=tmp_path / "safety.jsonl")


def test_allowed_commands_reach_the_drone(safe: SafeDrone, fake_drone) -> None:
    assert safe.takeoff(1.0).success
    assert safe.goto(1.0, 1.0, 1.5).success

    assert fake_drone.position == (1.0, 1.0, 1.5)
    assert [d.allowed for d in safe.decisions] == [True, True]


def test_refused_command_never_reaches_the_drone(safe: SafeDrone, fake_drone) -> None:
    safe.takeoff(1.0)
    before = (fake_drone.position, fake_drone.sim_time_s)

    result = safe.goto(0.0, 0.0, 50.0)

    assert result.refused and not result.success
    assert result.message.startswith("target z=50.0 m exceeds the 3.0 m altitude limit")
    assert result.position_m == before[0]
    assert (fake_drone.position, fake_drone.sim_time_s) == before


def test_move_relative_is_checked_as_a_goto(safe: SafeDrone, fake_drone) -> None:
    safe.takeoff(1.0)

    result = safe.move_relative(0.0, 5.0, 0.0)

    assert result.refused and "y=5.0 m is outside the geofence" in result.message
    assert fake_drone.position == (0.0, 0.0, 1.0)


def test_default_speed_is_checked_against_the_limit(fake_drone) -> None:
    fast_by_default = SafeDrone(fake_drone, default_speed_mps=2.0)
    fast_by_default.takeoff(1.0)

    assert fast_by_default.goto(1.0, 0.0, 1.0).refused


def test_low_battery_still_allows_landing(safe: SafeDrone, fake_drone, monkeypatch) -> None:
    safe.takeoff(1.0)
    low = DroneState((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 0.0, 5.0, True, fake_drone.sim_time_s)
    monkeypatch.setattr(fake_drone, "get_state", lambda: low)

    assert safe.goto(1.0, 0.0, 1.0).refused
    assert safe.land().success


def test_every_decision_is_logged_as_json_lines(safe: SafeDrone, tmp_path: Path) -> None:
    safe.takeoff(1.0)
    safe.goto(0.0, 0.0, 50.0)
    safe.land()

    lines = [json.loads(line) for line in (tmp_path / "safety.jsonl").read_text().splitlines()]
    assert [(r["command"], r["allowed"], r["rule"]) for r in lines] == [
        ("takeoff", True, "ok"),
        ("goto", False, "altitude"),
        ("land", True, "ok"),
    ]
    assert lines[1]["arguments"] == {"x_m": 0.0, "y_m": 0.0, "z_m": 50.0, "speed_mps": None}
    assert lines[0]["timestamp"].endswith("+00:00")


def test_mcp_tool_reports_refusal(safe: SafeDrone) -> None:
    server = build_server(safe)
    call = lambda name, **args: asyncio.run(server.call_tool(name, args))[0][0].text  # noqa: E731

    call("takeoff", altitude_m=1.0)
    text = call("goto", x_m=0.0, y_m=0.0, z_m=50.0)

    assert text.startswith(
        "REFUSED: target z=50.0 m exceeds the 3.0 m altitude limit; "
        "choose z between 0.2 and 3.0 m. Now flying at (0.00, 0.00, 1.00) m"
    )
