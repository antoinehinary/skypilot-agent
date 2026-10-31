from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from skypilot.config import CONFIG_DIR, SimConfig, load_sim_config


def _raw() -> dict:
    return yaml.safe_load((CONFIG_DIR / "sim.yaml").read_text())


def test_default_sim_config_loads() -> None:
    config = load_sim_config()

    assert config.physics_hz % config.control_hz == 0
    assert config.motion.position_tolerance_m == pytest.approx(0.1)


def test_unknown_key_is_rejected() -> None:
    raw = _raw()
    raw["motion"]["positon_tolerance_m"] = 0.2  # typo

    with pytest.raises(ValidationError, match="positon_tolerance_m"):
        SimConfig.model_validate(raw)


def test_incompatible_rates_are_rejected() -> None:
    raw = _raw()
    raw["control_hz"] = 50

    with pytest.raises(ValidationError, match="multiple"):
        SimConfig.model_validate(raw)


def test_load_from_explicit_path(tmp_path: Path) -> None:
    raw = _raw()
    raw["control_hz"] = 240
    path = tmp_path / "sim.yaml"
    path.write_text(yaml.safe_dump(raw))

    assert load_sim_config(path).control_hz == 240
