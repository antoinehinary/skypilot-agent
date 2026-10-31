"""Typed configuration loaded from the YAML files in `config/`.

Models forbid unknown keys, so a typo in a YAML file fails loudly at start-up
instead of silently falling back to a default.
"""

from __future__ import annotations

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


class SimConfig(_Model):
    physics_hz: PositiveInt
    control_hz: PositiveInt
    motion: MotionConfig
    battery: BatteryConfig
    gui: GuiConfig

    @model_validator(mode="after")
    def _check_rates(self) -> Self:
        if self.physics_hz % self.control_hz != 0:
            raise ValueError(
                f"physics_hz ({self.physics_hz}) must be a multiple of "
                f"control_hz ({self.control_hz})"
            )
        return self


def load_sim_config(path: Path | None = None) -> SimConfig:
    """Load and validate the simulator config (default: `config/sim.yaml`)."""
    path = path or CONFIG_DIR / "sim.yaml"
    with path.open() as f:
        return SimConfig.model_validate(yaml.safe_load(f))
