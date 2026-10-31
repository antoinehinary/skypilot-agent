"""PositionalArgsTool: positional calls on keyword-only MCP tools."""

import pytest
from smolagents import Tool

from skypilot.agent.tools import PositionalArgsTool


class EchoTool(Tool):
    name = "goto"
    description = "echo arguments"
    inputs = {
        "x_m": {"type": "number", "description": "x"},
        "y_m": {"type": "number", "description": "y"},
        "z_m": {"type": "number", "description": "z"},
    }
    output_type = "object"
    skip_forward_signature_validation = True

    def forward(self, **kwargs):  # keyword-only, like mcpadapt tools
        return kwargs


TOOL = PositionalArgsTool(EchoTool())


def test_keeps_name_description_and_inputs() -> None:
    assert (TOOL.name, TOOL.description, TOOL.inputs) == ("goto", "echo arguments", EchoTool.inputs)


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        ((1.0, 2.0, 3.0), {}),
        ((1.0,), {"z_m": 3.0, "y_m": 2.0}),
        ((), {"x_m": 1.0, "y_m": 2.0, "z_m": 3.0}),
    ],
)
def test_positional_arguments_map_to_inputs_in_order(args, kwargs) -> None:
    assert TOOL(*args, **kwargs) == {"x_m": 1.0, "y_m": 2.0, "z_m": 3.0}


def test_too_many_arguments_is_a_clear_error() -> None:
    with pytest.raises(TypeError, match=r"goto\(\) takes 3 argument\(s\) \(x_m, y_m, z_m\) but 4"):
        TOOL(1, 2, 3, 4)


def test_duplicate_argument_is_a_clear_error() -> None:
    with pytest.raises(TypeError, match="multiple values for argument 'x_m'"):
        TOOL(1.0, x_m=2.0)
