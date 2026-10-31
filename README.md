# SkyPilot Agent

An AI agent that flies a simulated Crazyflie from plain-language missions.
You type *"take off, find the red box, hover above it and land"*; the agent plans the mission,
calls drone tools over MCP, uses the onboard camera, and every command passes a safety layer
that can refuse it with a reason.

![Scripted square flight: overview with flight trail (left) and chase camera (right)](media/square.gif)

**Status:** early development. A local LLM agent flies plain-language missions through MCP tools;
vision and the safety layer are next.

## Roadmap

- [x] Drone backend: blocking takeoff / goto / hover / land on gym-pybullet-drones, scripted square flight
- [x] MCP server exposing the drone as tools + smolagents agent with a local LLM (Ollama)
- [ ] Vision: coloured-object detection and ground-plane localisation, scene description with a vision model
- [ ] Safety layer: geofence, altitude / speed / battery limits, structured refusals, CI
- [ ] Evaluation harness: mission suite, success metrics, model comparison table
- [ ] Demo GIFs, architecture diagram, v1.0

## Architecture

```
Mission (text) ──► Agent (smolagents + local LLM in Ollama)
                     │  MCP tool calls / results
                     ▼
                   MCP server (drone exposed as tools) ◄── Vision (OpenCV + Ollama vision model)
                     │
                     ▼
                   Safety layer (validates every action, refuses with a reason)
                     │  safe commands only
                     ▼
                   DroneBackend interface ──► SimBackend (gym-pybullet-drones)
                                         └──► (future) CrazyflieBackend (cflib)
```

## Setup (macOS / Linux, conda)

Everything (environment and simulator) is installed inside the repository folder.

```bash
git clone <this-repo> skypilot-agent && cd skypilot-agent
conda create -p ./.venv python=3.12 -y && conda activate ./.venv
git clone https://github.com/learnsyslab/gym-pybullet-drones.git external/gym-pybullet-drones
git -C external/gym-pybullet-drones checkout 15c426c
CFLAGS="-Dfdopen=fdopen" pip install -e ".[sim,agent,dev]" && pip install -e external/gym-pybullet-drones --no-deps
```

For the agent, install [Ollama](https://ollama.com), then `ollama pull qwen3:8b` and keep Ollama running.

The `CFLAGS` flag is needed to build `pybullet` from source on macOS. The simulator is installed with
`--no-deps` to skip its reinforcement-learning dependencies (torch, stable-baselines3), which this
project does not use; pip will warn about them, which is expected.

## Fly a mission with the agent

```bash
python -m skypilot.agent.run_agent "take off to 1 m, fly to (2, 2, 1), land" --gui
```

The agent ([smolagents](https://github.com/huggingface/smolagents) `CodeAgent`, `qwen3:8b` in Ollama by
default) starts the drone MCP server as a subprocess and talks to it over stdio. At each step the
model writes a short Python snippet that calls the drone tools, reads the result
(`OK: reached (2.00, 2.00, 1.00) in 5.7 s. Now flying at ...`) and decides the next step. It never
touches the simulator directly.

| Setting | Where |
| --- | --- |
| Model, context size, timeouts, step budget | [`config/agent.yaml`](config/agent.yaml) |
| Other model (any [LiteLLM](https://docs.litellm.ai/docs/providers) id) | `--model <id>` or `MODEL_ID` in the environment / `.env` (see `.env.example`) |

### Drone tools over MCP

`takeoff(altitude_m)`, `goto(x_m, y_m, z_m)`, `move_relative(dx_m, dy_m, dz_m)`, `hover(seconds)`,
`land()`, `get_state()`. The server works with any MCP client; to explore it in the
[MCP Inspector](https://github.com/modelcontextprotocol/inspector):

```bash
npx @modelcontextprotocol/inspector ./.venv/bin/python -m skypilot.mcp_server.server --gui
```

## Scripted flight

```bash
python scripts/square_flight.py --gui   # 1 m square at 1 m altitude, saves media/square.gif
python scripts/square_flight.py --help  # side, altitude, speed, GIF options
```

The drone follows a straight-line setpoint at the requested speed with velocity feed-forward
into the cascaded PID controller from gym-pybullet-drones (`DSLPIDControl`). Each command blocks
until the drone is within 0.1 m of its target or a timeout expires. Parameters live in
[`config/sim.yaml`](config/sim.yaml).

## Tests

```bash
python -m pytest               # everything, simulator headless, LLM replaced by a scripted model (~5 s)
python -m pytest -m "not sim"  # pure-Python tests only, no PyBullet needed
python -m ruff check . && python -m ruff format --check .
```

## Conventions

- SI units everywhere: metres, seconds, m/s. Degrees only in user-facing text.
- World frame: **x forward, y left, z up**, origin at the take-off pad.

## License

MIT
