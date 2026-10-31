"""Tests of the MCP server: in-process against FakeBackend, plus one real stdio round trip."""

import asyncio
import re
import sys

import numpy as np
import pytest
from mcp.server.fastmcp.exceptions import ToolError

from skypilot.drone.backend import DroneState, MoveResult
from skypilot.mcp_server.server import build_server, format_result
from skypilot.vision import describe
from skypilot.vision.geometry import world_to_pixel

MOVE_TOOLS = {"takeoff", "goto", "move_relative", "hover", "land", "get_state"}
CAMERA_TOOLS = {"get_camera_image", "find_objects", "describe_scene"}
EXPECTED_TOOLS = MOVE_TOOLS | CAMERA_TOOLS


def _call(server, name: str, **arguments) -> str:
    content, _ = asyncio.run(server.call_tool(name, arguments))
    return content[0].text


@pytest.fixture
def server(fake_drone, tmp_path):
    return build_server(fake_drone, image_dir=tmp_path / "images")


def _image_with_red_square(fake_drone, ground_xy: tuple[float, float]) -> None:
    """Paint a red 0.3 m square where `ground_xy` appears in the fake drone's camera."""
    frame = fake_drone.get_camera_image()
    u, v = world_to_pixel(np.array([*ground_xy, 0.0]), frame.intrinsics, frame.cam_to_world)
    half = round(0.15 * frame.intrinsics[0, 0] / fake_drone.position[2])
    image = np.full((240, 320, 3), 224, np.uint8)
    image[round(v) - half : round(v) + half, round(u) - half : round(u) + half] = (210, 25, 25)
    fake_drone.camera_image = image


def test_find_objects_reports_world_position_and_refine_hint(server, fake_drone) -> None:
    _call(server, "takeoff", altitude_m=2.0)
    _image_with_red_square(fake_drone, (0.6, -0.4))

    text = _call(server, "find_objects", colour="Red")

    match = re.match(
        r"Found 1 red object: red #1 at \(([-\d.]+), ([-\d.]+)\) m, about ([\d.]+)", text
    )
    assert match, text
    x, y, size = map(float, match.groups())
    assert (x, y) == pytest.approx((0.6, -0.4), abs=0.015)  # painted square is pixel-quantised
    assert size == pytest.approx(0.3, abs=0.03)
    assert "find_objects again to refine" in text


def test_find_objects_not_found_explains_coverage(server) -> None:
    _call(server, "takeoff", altitude_m=2.0)

    text = _call(server, "find_objects", colour="red")

    assert text.startswith("No red object in view")
    assert "altitude 2.00 m" in text and "Climb" in text


def test_find_objects_unknown_colour_lists_options(server) -> None:
    assert _call(server, "find_objects", colour="purple").startswith(
        "Unknown colour 'purple'. Available colours: blue, green, red, yellow."
    )


def test_get_camera_image_saves_png(server, fake_drone, tmp_path) -> None:
    _call(server, "takeoff", altitude_m=1.0)

    text = _call(server, "get_camera_image")

    saved = list((tmp_path / "images").glob("*.png"))
    assert len(saved) == 1
    assert text.startswith(f"Saved 320x240 image to {saved[0]}")


def test_describe_scene_returns_model_answer_or_readable_failure(server, monkeypatch) -> None:
    monkeypatch.setattr(describe, "describe_scene", lambda frame, q, cfg: f"answer to: {q}")
    assert _call(server, "describe_scene", question="clear?") == "answer to: clear?"

    def unavailable(*_):
        raise describe.VisionModelError("vision model x unavailable; is Ollama running?")

    monkeypatch.setattr(describe, "describe_scene", unavailable)
    assert _call(server, "describe_scene", question="clear?").startswith("FAILED: vision model")


def test_exposes_expected_tools(server) -> None:
    tools = asyncio.run(server.list_tools())

    assert {t.name for t in tools} == EXPECTED_TOOLS


def test_every_tool_and_argument_is_described(server) -> None:
    for tool in asyncio.run(server.list_tools()):
        assert tool.description, tool.name
        for arg, schema in tool.inputSchema["properties"].items():
            assert schema.get("description"), f"{tool.name}.{arg}"


def test_tools_that_do_not_move_the_drone_are_read_only(server) -> None:
    tools = {t.name: t for t in asyncio.run(server.list_tools())}

    for name in {"get_state"} | CAMERA_TOOLS:
        assert tools[name].annotations.readOnlyHint, name
    for name in MOVE_TOOLS - {"get_state"}:
        assert not (tools[name].annotations and tools[name].annotations.readOnlyHint), name


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
