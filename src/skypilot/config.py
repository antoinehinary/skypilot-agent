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
    position_tolerance_m: PositiveFloat
    default_speed_mps: PositiveFloat
    takeoff_speed_mps: PositiveFloat
    landing_speed_mps: PositiveFloat
    settle_timeout_s: PositiveFloat
    motor_cutoff_height_m: PositiveFloat


class BatteryConfig(_Model):
    initial_pct: PositiveFloat
    hover_drain_pct_per_s: float
    speed_drain_pct_per_mps: float


class GuiConfig(_Model):
    realtime: bool
    camera_distance_m: PositiveFloat
    camera_yaw_deg: float
    camera_pitch_deg: float


class CameraConfig(_Model):
    width_px: PositiveInt
    height_px: PositiveInt
    vertical_fov_deg: PositiveFloat
    mount_offset_m: float
    near_m: PositiveFloat
    far_m: PositiveFloat


class SimConfig(_Model):
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
    num_ctx: PositiveInt
    think: bool


class AgentConfig(_Model):
    model_id: str
    api_base: str
    request_timeout_s: PositiveFloat
    temperature: float
    max_steps: PositiveInt
    ollama: OllamaConfig


class WorldObject(_Model):
    name: str
    colour: str
    position_m: tuple[float, float]
    size_m: PositiveFloat


class WorldConfig(_Model):
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


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def load_sim_config(path: Path | None = None) -> SimConfig:
    """Load and validate the simulator config (default: `config/sim.yaml`)."""
    return SimConfig.model_validate(_load_yaml(path or CONFIG_DIR / "sim.yaml"))


def load_world_config(path: Path | None = None) -> WorldConfig:
    """Load the scene objects (default: `config/world.yaml`)."""
    return WorldConfig.model_validate(_load_yaml(path or CONFIG_DIR / "world.yaml"))


def load_agent_config(path: Path | None = None) -> AgentConfig:
    """Load the agent config (default: `config/agent.yaml`).

    The MODEL_ID and OLLAMA_API_BASE environment variables override `model_id` and `api_base`.
    """
    raw = _load_yaml(path or CONFIG_DIR / "agent.yaml")
    for key, env in (("model_id", "MODEL_ID"), ("api_base", "OLLAMA_API_BASE")):
        if os.environ.get(env):
            raw[key] = os.environ[env]
    return AgentConfig.model_validate(raw)
