"""Fly a plain-language mission with an LLM agent connected to the drone MCP server.

Usage:
    python -m skypilot.agent.run_agent "take off to 1 m, fly to (2, 2, 1), land" --gui
    python -m skypilot.agent.run_agent "..." --model ollama_chat/qwen3:14b

The agent never touches the simulator: it starts the MCP server as a subprocess and
calls its tools over stdio.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from dotenv import load_dotenv
from mcp import StdioServerParameters
from smolagents import CodeAgent, LiteLLMModel, LogLevel, MCPClient, Model
from smolagents.agents import RunResult

from skypilot.agent.prompts import INSTRUCTIONS
from skypilot.config import AgentConfig, load_agent_config

logger = logging.getLogger(__name__)


def build_model(config: AgentConfig) -> LiteLLMModel:
    """LiteLLM client for `config.model_id`, with Ollama-specific options when relevant."""
    kwargs: dict = {"timeout": config.request_timeout_s, "temperature": config.temperature}
    if config.model_id.startswith("ollama"):
        kwargs |= {"num_ctx": config.ollama.num_ctx, "think": config.ollama.think}
        api_base = config.api_base
    else:
        api_base = None  # cloud providers: LiteLLM picks the endpoint and reads the API key
    return LiteLLMModel(model_id=config.model_id, api_base=api_base, **kwargs)


def server_parameters(gui: bool = False) -> StdioServerParameters:
    """Command that starts the drone MCP server with the current interpreter.

    The server shares this terminal's stderr; its logs are limited to warnings because
    the agent already prints every tool result.
    """
    args = ["-m", "skypilot.mcp_server.server", "--log-level", "WARNING"]
    if gui:
        args.append("--gui")
    return StdioServerParameters(command=sys.executable, args=args)


def count_action_steps(result: RunResult) -> int:
    """Number of agent action steps (excludes the task and planning entries)."""
    return sum(1 for step in result.steps if "step_number" in step)


def run_mission(
    mission: str,
    model: Model,
    max_steps: int,
    gui: bool = False,
    verbosity: LogLevel = LogLevel.INFO,
) -> RunResult:
    """Start the MCP server, run the agent on `mission`, and return the full run result."""
    with MCPClient(server_parameters(gui), structured_output=False) as tools:
        agent = CodeAgent(
            tools=tools,
            model=model,
            instructions=INSTRUCTIONS,
            max_steps=max_steps,
            code_block_tags="markdown",
            verbosity_level=verbosity,
        )
        return agent.run(mission, return_full_result=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fly a plain-language drone mission.")
    parser.add_argument("mission", help='e.g. "take off to 1 m, fly to (2, 2, 1), land"')
    parser.add_argument("--gui", action="store_true", help="show the PyBullet GUI")
    parser.add_argument("--model", help="LiteLLM model id (default: MODEL_ID or config/agent.yaml)")
    parser.add_argument("--max-steps", type=int, help="override max_steps from config")
    parser.add_argument("--quiet", action="store_true", help="only print the final report")
    args = parser.parse_args(argv)

    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    config = load_agent_config()
    if args.model:
        config = config.model_copy(update={"model_id": args.model})

    print(f"Model: {config.model_id}")
    start = time.monotonic()
    result = run_mission(
        args.mission,
        build_model(config),
        max_steps=args.max_steps or config.max_steps,
        gui=args.gui,
        verbosity=LogLevel.OFF if args.quiet else LogLevel.INFO,
    )
    tokens = result.token_usage.total_tokens if result.token_usage else "n/a"
    print(f"\n=== Mission report ({result.state}) ===\n{result.output}")
    print(
        f"\nSteps: {count_action_steps(result)} | tokens: {tokens} | "
        f"wall time: {time.monotonic() - start:.1f} s"
    )
    return 0 if result.state == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
