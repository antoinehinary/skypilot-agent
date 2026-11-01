# Changelog

## 1.0.0 (2026-11-01)

First release: an LLM agent that flies a simulated Crazyflie from plain-language missions, through
MCP tools and a safety layer, with an evaluation harness. Everything runs locally.

**Drone**
- `DroneBackend` interface with blocking takeoff, goto, move_relative, hover and land; relative
  moves offset from the commanded setpoint, not the drifting measured position.
- `SimBackend` on gym-pybullet-drones (Crazyflie 2.x, DSL cascaded PID): straight-line setpoint
  ramps with velocity feed-forward, contact-based landing (also on top of objects), linear
  battery model.
- Calibrated downward camera: each `CameraFrame` carries its intrinsics and camera-to-world pose.
- Scripted square flight and two-panel GIF recording of any flight.

**Agent and tools**
- FastMCP server over stdio: flight tools, `get_state`, `find_objects`, `get_camera_image`,
  `describe_scene`. Results end with the next call to make (fly above a detection, next search
  view, how to change altitude).
- smolagents `CodeAgent` with a local model in Ollama (`qwen3:8b` by default, any LiteLLM model id
  configurable); prints the drone's verified final state next to the agent's report.

**Vision**
- HSV colour detection with thresholds measured on rendered frames, ground-plane localisation by
  ray casting, search planning that tiles the geofence with camera footprints.
- Open questions answered by a vision-language model (`qwen2.5vl:7b`).

**Safety**
- `SafeDrone` wraps any backend: geofence, altitude band, speed, battery thresholds and mission
  time, as pure rules. Refusals give the reason and the nearest allowed command; landing is never
  refused. Every decision is logged as JSON Lines.

**Evaluation**
- 10 missions (basic, vision, safety) in YAML with ground-truth checks and reference solutions
  replayed in CI. Runner, per-mission transcripts and safety logs, and a generated README table
  comparing four local configurations.

**Tooling**
- 176 tests (pure-Python and headless simulator, LLMs replaced by scripted models), ruff with
  docstring checks, GitHub Actions CI.
