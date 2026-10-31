"""Agent tests without an LLM: a scripted model drives the real agent, MCP server and simulator."""

import math
import re
from typing import Any

import pytest
from smolagents import LogLevel, Model
from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import TokenUsage

from skypilot.agent.run_agent import (
    build_model,
    count_action_steps,
    run_mission,
    server_parameters,
)
from skypilot.config import load_agent_config


class ScriptedModel(Model):
    """Returns pre-written replies in order and records every prompt it receives."""

    def __init__(self, replies: list[str]) -> None:
        super().__init__(model_id="scripted")
        self.replies = list(replies)
        self.prompts: list[str] = []

    def generate(self, messages: list, stop_sequences: Any = None, **kwargs: Any) -> ChatMessage:
        self.prompts.append(str(messages))
        reply = self.replies.pop(0) if self.replies else "Thought: out of script.\n"
        return ChatMessage(role=MessageRole.ASSISTANT, content=reply, token_usage=TokenUsage(10, 5))


def _code(source: str) -> str:
    return f"Thought: next step.\n```python\n{source}\n```"


# ---------------------------------------------------------------- configuration


def test_model_id_can_be_overridden_by_environment(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_ID", "openai/gpt-4o-mini")

    assert load_agent_config().model_id == "openai/gpt-4o-mini"


def test_ollama_options_only_sent_to_ollama_models() -> None:
    config = load_agent_config()
    ollama = build_model(config)
    cloud = build_model(config.model_copy(update={"model_id": "openai/gpt-4o-mini"}))

    assert ollama.kwargs["num_ctx"] == config.ollama.num_ctx
    assert ollama.kwargs["think"] is False
    assert ollama.api_base == config.api_base
    assert "num_ctx" not in cloud.kwargs and "think" not in cloud.kwargs
    assert cloud.api_base is None


def test_server_parameters_forward_gui_flag() -> None:
    assert "--gui" in server_parameters(gui=True).args
    assert "--gui" not in server_parameters(gui=False).args


# ------------------------------------------------------- end to end, scripted


@pytest.fixture
def _needs_sim() -> None:
    pytest.importorskip("pybullet")


@pytest.mark.sim
@pytest.mark.usefixtures("_needs_sim")
def test_scripted_mission_flies_through_mcp() -> None:
    model = ScriptedModel(
        [
            _code("print(takeoff(altitude_m=1.0))"),
            _code("print(goto(x_m=2.0, y_m=2.0, z_m=1.0))"),
            _code("print(land())"),
            _code("final_answer(get_state())"),
        ]
    )

    result = run_mission("fly to (2, 2, 1)", model, max_steps=6, verbosity=LogLevel.OFF)

    assert result.state == "success"
    assert result.output.startswith("landed at (2.00, 2.00, 0.01) m")
    assert count_action_steps(result) == 4


@pytest.mark.sim
@pytest.mark.usefixtures("_needs_sim")
def test_failed_tool_result_reaches_the_model() -> None:
    model = ScriptedModel(
        [
            _code("print(goto(x_m=1.0, y_m=0.0, z_m=1.0))"),  # not flying yet
            _code("final_answer('gave up')"),
        ]
    )

    run_mission("fly to (1, 0, 1)", model, max_steps=4, verbosity=LogLevel.OFF)

    assert "FAILED: not flying: take off first" in model.prompts[1]


@pytest.mark.sim
@pytest.mark.usefixtures("_needs_sim")
def test_agent_stops_at_step_budget() -> None:
    model = ScriptedModel([_code("print(get_state())")] * 10)

    result = run_mission("hover forever", model, max_steps=2, verbosity=LogLevel.OFF)

    assert result.state == "max_steps_error"
    # 2 code steps + 1 forced final-answer call made by smolagents when the budget runs out.
    assert count_action_steps(result) == 3


FLY_TO_RED = """
import re
found = find_objects(colour="red")
print(found)
x, y = map(float, re.search(r"at \\(([-\\d.]+), ([-\\d.]+)\\)", found).groups())
print(goto(x_m=x, y_m=y, z_m={z}))
"""


@pytest.mark.sim
@pytest.mark.usefixtures("_needs_sim")
def test_scripted_search_finds_and_hovers_above_the_red_box() -> None:
    from skypilot.config import load_world_config

    red = next(o for o in load_world_config().objects if o.colour == "red")
    model = ScriptedModel(
        [
            _code("print(takeoff(altitude_m=2.5))"),
            _code(FLY_TO_RED.format(z=2.5)),  # coarse estimate from 1.8 m away
            _code(FLY_TO_RED.format(z=1.0)),  # refined from directly above, then descend
            _code("print(hover(seconds=2.0))"),
            _code("final_answer(get_state())"),
        ]
    )

    result = run_mission("find the red box", model, max_steps=8, verbosity=LogLevel.OFF)

    assert result.state == "success", result.output
    x, y, z = map(
        float, re.search(r"at \(([-\d.]+), ([-\d.]+), ([-\d.]+)\)", result.output).groups()
    )
    assert math.dist((x, y), red.position_m) < 0.05
    assert z == pytest.approx(1.0, abs=0.1)


@pytest.mark.sim
@pytest.mark.usefixtures("_needs_sim")
def test_positional_tool_calls_work_through_mcp() -> None:
    model = ScriptedModel([_code("print(takeoff(1.0))"), _code("final_answer(get_state())")])

    result = run_mission("take off", model, max_steps=4, verbosity=LogLevel.OFF)

    match = re.search(r"flying at \(([-\d.]+), ([-\d.]+), ([-\d.]+)\)", result.output)
    assert match, result.output
    assert float(match.group(3)) == pytest.approx(1.0, abs=0.1)
