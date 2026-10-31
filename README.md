# SkyPilot Agent

An AI agent that flies a simulated Crazyflie from plain-language missions.
You type *"take off, find the red box, hover above it and land"*; the agent plans the mission,
calls drone tools over MCP, uses the onboard camera, and every command passes a safety layer
that can refuse it with a reason.

**Status:** early development. The simulator backend is being built; the agent is not usable yet.

## Roadmap

- [ ] Drone backend: blocking takeoff / goto / hover / land on gym-pybullet-drones, scripted square flight
- [ ] MCP server exposing the drone as tools + smolagents agent with a local LLM (Ollama)
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
CFLAGS="-Dfdopen=fdopen" pip install -e ".[sim,dev]" && pip install -e external/gym-pybullet-drones --no-deps
```

The `CFLAGS` flag is needed to build `pybullet` from source on macOS. The simulator is installed with
`--no-deps` to skip its reinforcement-learning dependencies (torch, stable-baselines3), which this
project does not use; pip will warn about them, which is expected.

Check: `pytest` and `ruff check .` should pass.

## Conventions

- SI units everywhere: metres, seconds, m/s. Degrees only in user-facing text.
- World frame: **x forward, y left, z up**, origin at the take-off pad.

## License

MIT
