"""Typed configuration loaded from the YAML files in `config/`.

Models forbid unknown keys, so a typo in a YAML file fails loudly at start-up
instead of silently falling back to a default.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt, model_validator

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MotionConfig(_Model):
    """Move tolerances, speeds and timeouts (sim.yaml `motion`)."""

    position_tolerance_m: PositiveFloat
    default_speed_mps: PositiveFloat
    takeoff_speed_mps: PositiveFloat
    landing_speed_mps: PositiveFloat
    settle_timeout_s: PositiveFloat
    motor_cutoff_height_m: PositiveFloat


class BatteryConfig(_Model):
    """Simulated battery drain model (sim.yaml `battery`)."""

    initial_pct: PositiveFloat
    hover_drain_pct_per_s: float
    speed_drain_pct_per_mps: float


class GuiConfig(_Model):
    """GUI pacing and orbit camera (sim.yaml `gui`)."""

    realtime: bool
    camera_distance_m: PositiveFloat
    camera_yaw_deg: float
    camera_pitch_deg: float


class CameraConfig(_Model):
    """Downward onboard camera (sim.yaml `camera`)."""

    width_px: PositiveInt
    height_px: PositiveInt
    vertical_fov_deg: PositiveFloat
    mount_offset_m: float
    near_m: PositiveFloat
    far_m: PositiveFloat


class SimConfig(_Model):
    """Simulator parameters (config/sim.yaml)."""

    physics_hz: PositiveInt
    control_hz: PositiveInt
    motion: MotionConfig
    battery: BatteryConfig
    camera: CameraConfig
    gui: GuiConfig

    @model_validator(mode="after")
    def _check_rates(self) -> Self:
        if self.physics_hz % self.control_hz != 0:
            raise ValueError(
                f"physics_hz ({self.physics_hz}) must be a multiple of "
                f"control_hz ({self.control_hz})"
            )
        return self


class OllamaConfig(_Model):
    """Options sent only to Ollama models (agent.yaml `ollama`)."""

    num_ctx: PositiveInt
    think: bool


class AgentConfig(_Model):
    """Agent and LLM parameters (config/agent.yaml)."""

    model_id: str
    api_base: str
    request_timeout_s: PositiveFloat
    max_tokens: PositiveInt | None
    temperature: float
    seed: int | None
    max_steps: PositiveInt
    planning_interval: PositiveInt | None
    ollama: OllamaConfig


class WorldObject(_Model):
    """One box in the scene: name, colour, ground position (m), side (m)."""

    name: str
    colour: str
    position_m: tuple[float, float]
    size_m: PositiveFloat


class WorldConfig(_Model):
    """Scene objects and their rendering colours (config/world.yaml)."""

    colours: dict[str, tuple[float, float, float, float]]
    objects: list[WorldObject]

    @model_validator(mode="after")
    def _check_colours(self) -> Self:
        for obj in self.objects:
            if obj.colour not in self.colours:
                raise ValueError(
                    f"object {obj.name!r} uses colour {obj.colour!r}, "
                    f"not one of {sorted(self.colours)}"
                )
        return self


class HsvRange(_Model):
    """Inclusive OpenCV HSV bounds (hue 0-179, saturation and value 0-255)."""

    low: tuple[int, int, int]
    high: tuple[int, int, int]


class DescribeConfig(_Model):
    """Vision-language model used by describe_scene (vision.yaml `describe`)."""

    model_id: str
    api_base: str
    request_timeout_s: PositiveFloat
    max_tokens: PositiveInt


class VisionConfig(_Model):
    """Colour thresholds and scene-description model (config/vision.yaml)."""

    colours: dict[str, list[HsvRange]]
    min_area_px: PositiveInt
    describe: DescribeConfig


class Interval(_Model):
    """Closed interval [low, high], written as a two-element list in YAML."""

    low: float
    high: float

    @model_validator(mode="before")
    @classmethod
    def _from_pair(cls, value: object) -> object:
        if isinstance(value, list | tuple) and len(value) == 2:
            return {"low": value[0], "high": value[1]}
        return value

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.low > self.high:
            raise ValueError(f"interval [{self.low}, {self.high}] is empty")
        return self

    def __contains__(self, value: float) -> bool:
        return self.low <= value <= self.high


class GeofenceConfig(_Model):
    """Allowed x and y ranges, metres (safety.yaml `geofence`)."""

    x_m: Interval
    y_m: Interval


class SafetyBatteryConfig(_Model):
    """Battery thresholds, percent (safety.yaml `battery`)."""

    min_takeoff_pct: PositiveFloat
    reserve_pct: PositiveFloat


class SafetyConfig(_Model):
    """Safety limits enforced by SafeDrone (config/safety.yaml)."""

    geofence: GeofenceConfig
    altitude_m: Interval
    max_horizontal_speed_mps: PositiveFloat
    battery: SafetyBatteryConfig
    max_mission_time_s: PositiveFloat


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def load_sim_config(path: Path | None = None) -> SimConfig:
    """Load and validate the simulator config (default: `config/sim.yaml`)."""
    return SimConfig.model_validate(_load_yaml(path or CONFIG_DIR / "sim.yaml"))


def load_world_config(path: Path | None = None) -> WorldConfig:
    """Load the scene objects (default: `config/world.yaml`)."""
    return WorldConfig.model_validate(_load_yaml(path or CONFIG_DIR / "world.yaml"))


def load_vision_config(path: Path | None = None) -> VisionConfig:
    """Load colour thresholds and the scene-description model (default: `config/vision.yaml`)."""
    return VisionConfig.model_validate(_load_yaml(path or CONFIG_DIR / "vision.yaml"))


def load_safety_config(path: Path | None = None) -> SafetyConfig:
    """Load the safety limits (default: `config/safety.yaml`)."""
    return SafetyConfig.model_validate(_load_yaml(path or CONFIG_DIR / "safety.yaml"))


def load_agent_config(path: Path | None = None) -> AgentConfig:
    """Load the agent config (default: `config/agent.yaml`).

    The MODEL_ID and OLLAMA_API_BASE environment variables override `model_id` and `api_base`.
    """
    raw = _load_yaml(path or CONFIG_DIR / "agent.yaml")
    for key, env in (("model_id", "MODEL_ID"), ("api_base", "OLLAMA_API_BASE")):
        if os.environ.get(env):
            raw[key] = os.environ[env]
    return AgentConfig.model_validate(raw)
