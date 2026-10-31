"""Tests of the MCP server: in-process against FakeBackend, plus one real stdio round trip."""

import asyncio
import sys

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from skypilot.drone.backend import DroneState, MoveResult
from skypilot.mcp_server.server import build_server, format_result

EXPECTED_TOOLS = {"takeoff", "goto", "move_relative", "hover", "land", "get_state"}


def _call(server, name: str, **arguments) -> str:
    content, _ = asyncio.run(server.call_tool(name, arguments))
    return content[0].text


@pytest.fixture
def server(fake_drone):
    return build_server(fake_drone)


def test_exposes_expected_tools(server) -> None:
    tools = asyncio.run(server.list_tools())

    assert {t.name for t in tools} == EXPECTED_TOOLS


def test_every_tool_and_argument_is_described(server) -> None:
    for tool in asyncio.run(server.list_tools()):
        assert tool.description, tool.name
        for arg, schema in tool.inputSchema["properties"].items():
            assert schema.get("description"), f"{tool.name}.{arg}"


def test_get_state_is_read_only(server) -> None:
    tools = {t.name: t for t in asyncio.run(server.list_tools())}

    assert tools["get_state"].annotations.readOnlyHint


def test_move_tools_report_status_and_resulting_state(server) -> None:
    text = _call(server, "takeoff", altitude_m=1.0)

    assert text.startswith("OK: ")
    assert "flying at (0.00, 0.00, 1.00) m" in text
    assert "battery 100.0 %" in text


def test_move_relative_and_get_state(server) -> None:
    _call(server, "takeoff", altitude_m=1.0)
    _call(server, "move_relative", dx_m=1.0, dy_m=-0.5, dz_m=0.0)

    assert _call(server, "get_state").startswith("flying at (1.00, -0.50, 1.00) m")


def test_failed_move_is_reported_as_failed() -> None:
    result = MoveResult(False, (0.0, 0.0, 0.0), 0.0, "not flying: take off first")
    state = DroneState((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, 100.0, False, 0.0)

    text = format_result(result, state)

    assert text.startswith("FAILED: not flying: take off first. Now landed at")


@pytest.mark.parametrize(
    ("tool", "arguments", "expected"),
    [
        ("hover", {"seconds": -1.0}, "greater than 0"),
        ("takeoff", {"altitude_m": "high"}, "valid number"),
        ("goto", {"x_m": 1.0}, "Field required"),
    ],
)
def test_invalid_arguments_are_rejected_before_reaching_the_drone(
    server, fake_drone, tool, arguments, expected
) -> None:
    with pytest.raises(ToolError, match=expected):
        _call(server, tool, **arguments)

    assert fake_drone.sim_time_s == 0.0


@pytest.mark.sim
def test_stdio_round_trip_with_the_real_simulator() -> None:
    pytest.importorskip("pybullet")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "skypilot.mcp_server.server"]
    )

    async def session_run() -> list[str]:
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert {t.name for t in tools.tools} == EXPECTED_TOOLS
            texts = []
            for name, args in [("takeoff", {"altitude_m": 1.0}), ("land", {})]:
                result = await session.call_tool(name, args)
                assert not result.isError
                texts.append(result.content[0].text)
            return texts

    takeoff, land = asyncio.run(asyncio.wait_for(session_run(), timeout=60))

    assert takeoff.startswith("OK: took off to (0.00, 0.00, 1.00)")
    assert land.startswith("OK: landed at")
