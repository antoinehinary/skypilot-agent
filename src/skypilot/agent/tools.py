"""Adapters between MCP tools and the code the agent writes."""

from __future__ import annotations

from typing import Any, override

from smolagents import Tool


class PositionalArgsTool(Tool):
    """Wrap a tool so it also accepts positional arguments, mapped to its inputs in order.

    MCP tools bridged by mcpadapt only accept keyword arguments, so `takeoff(2.5)` fails with
    a confusing error, and small models then tend to repeat the same call. With this wrapper
    `takeoff(2.5)` means `takeoff(altitude_m=2.5)`, as it would for a normal Python function.
    """

    skip_forward_signature_validation = True

    def __init__(self, tool: Tool) -> None:
        self.name = tool.name
        self.description = tool.description
        self.inputs = tool.inputs
        self.output_type = tool.output_type
        self.output_schema = getattr(tool, "output_schema", None)
        self.is_initialized = True
        self._tool = tool

    @override
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        names = list(self.inputs)
        if len(args) > len(names):
            raise TypeError(
                f"{self.name}() takes {len(names)} argument(s) ({', '.join(names) or 'none'}) "
                f"but {len(args)} were given"
            )
        for name, value in zip(names, args, strict=False):
            if name in kwargs:
                raise TypeError(f"{self.name}() got multiple values for argument '{name}'")
            kwargs[name] = value
        return self._tool(**kwargs)
