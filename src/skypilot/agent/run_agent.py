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
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from mcp import StdioServerParameters
from smolagents import CodeAgent, LiteLLMModel, LogLevel, MCPClient, Model
from smolagents.agents import RunResult

from skypilot.agent.prompts import INSTRUCTIONS
from skypilot.agent.tools import PositionalArgsTool
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


@dataclass(frozen=True)
class ServerOptions:
    """How to start the drone MCP server for one mission.

    Attributes:
        gui: Show the PyBullet GUI.
        world: Scene objects YAML; None for config/world.yaml.
        initial_battery_pct: Starting battery; None for the sim config value.
        safety_log: JSON Lines file for this run's safety decisions; None for the default.
        log_level: Server log level on stderr (WARNING shows safety refusals).
    """

    gui: bool = False
    world: Path | None = None
    initial_battery_pct: float | None = None
    safety_log: Path | None = None
    log_level: str = "WARNING"


def server_parameters(options: ServerOptions | None = None) -> StdioServerParameters:
    """Command that starts the drone MCP server with the current interpreter.

    The server shares this terminal's stderr; its logs are limited to warnings because
    the agent already prints every tool result.
    """
    options = options or ServerOptions()
    args = ["-m", "skypilot.mcp_server.server", "--log-level", options.log_level]
    if options.gui:
        args.append("--gui")
    if options.world is not None:
        args += ["--world", str(options.world)]
    if options.initial_battery_pct is not None:
        args += ["--initial-battery", str(options.initial_battery_pct)]
    if options.safety_log is not None:
        args += ["--safety-log", str(options.safety_log)]
    return StdioServerParameters(command=sys.executable, args=args)


def count_action_steps(result: RunResult) -> int:
    """Number of agent action steps, i.e. LLM calls that produced an action.

    Excludes the task and planning entries. When the step budget runs out, smolagents
    makes one extra LLM call to write a final answer; it is counted too.
    """
    return sum(1 for step in result.steps if "step_number" in step)


@dataclass(frozen=True)
class MissionResult:
    """Agent run plus the drone's actual state afterwards.

    Attributes:
        run: smolagents run result (report in `run.output`, steps, tokens, state).
        final_state: Drone state read from the get_state tool after the agent finished,
            independently of the LLM: the ground truth to check the report against.
    """

    run: RunResult
    final_state: str


def run_mission(
    mission: str,
    model: Model,
    max_steps: int,
    server: ServerOptions | None = None,
    verbosity: LogLevel = LogLevel.INFO,
    planning_interval: int | None = None,
) -> MissionResult:
    """Start the MCP server, run the agent on `mission`, then read the drone's final state.

    `planning_interval` makes smolagents write an explicit plan every N steps (None: never).
    """
    with MCPClient(server_parameters(server), structured_output=False) as tools:
        agent = CodeAgent(
            tools=[PositionalArgsTool(t) for t in tools],
            model=model,
            instructions=INSTRUCTIONS,
            max_steps=max_steps,
            planning_interval=planning_interval,
            code_block_tags="markdown",
            verbosity_level=verbosity,
        )
        run = agent.run(mission, return_full_result=True)
        get_state = next(t for t in tools if t.name == "get_state")
        return MissionResult(run=run, final_state=str(get_state()))


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
    mission = run_mission(
        args.mission,
        build_model(config),
        max_steps=args.max_steps or config.max_steps,
        server=ServerOptions(gui=args.gui),
        verbosity=LogLevel.OFF if args.quiet else LogLevel.INFO,
        planning_interval=config.planning_interval,
    )
    run = mission.run
    tokens = run.token_usage.total_tokens if run.token_usage else "n/a"
    print(f"\n=== Mission report from the agent ({run.state}) ===\n{run.output}")
    print(f"\n=== Verified final state (read from the drone) ===\n{mission.final_state}")
    print(
        f"\nSteps: {count_action_steps(run)} | tokens: {tokens} | "
        f"wall time: {time.monotonic() - start:.1f} s"
    )
    return 0 if run.state == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
