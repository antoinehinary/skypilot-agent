"""Mission definitions loaded from `missions/*.yaml`.

A mission is a plain-language task, an optional setup (scene, starting battery), the
checks that decide success, and a reference solution: the tool calls a perfect agent
would make, replayed by a scripted model to prove the mission and its checks are sound.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

MISSIONS_DIR = Path(__file__).resolve().parents[3] / "missions"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FinalPosition(_Model):
    """Final (x, y) within `tolerance_m` of a point or of a world object; optional z range."""

    type: Literal["final_position"]
    xy: tuple[float, float] | None = None
    object: str | None = None
    tolerance_m: PositiveFloat
    z_m: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _one_target(self) -> Self:
        if (self.xy is None) == (self.object is None):
            raise ValueError("final_position needs exactly one of `xy` or `object`")
        return self


class Landed(_Model):
    """The drone ends on the ground (True) or in the air (False)."""

    type: Literal["landed"]
    expected: bool = True


class Visited(_Model):
    """Every point was the target of an accepted goto (from the safety log), within tolerance."""

    type: Literal["visited"]
    points: list[tuple[float, float]]
    tolerance_m: PositiveFloat


class Refused(_Model):
    """The safety layer refused at least one command under `rule`."""

    type: Literal["refused"]
    rule: str


class NoRefusals(_Model):
    """No command was refused (the mission is within limits: no false refusals)."""

    type: Literal["no_refusals"]


class NeverTookOff(_Model):
    """No takeoff was accepted."""

    type: Literal["never_took_off"]


class ReportMentions(_Model):
    """The agent's final report contains at least one of the phrases (case-insensitive)."""

    type: Literal["report_mentions"]
    any: list[str]


Check = Annotated[
    FinalPosition | Landed | Visited | Refused | NoRefusals | NeverTookOff | ReportMentions,
    Field(discriminator="type"),
]


class Setup(_Model):
    """Mission setup: alternative world file and starting battery."""

    world: Path | None = None  # relative to the missions directory
    initial_battery_pct: PositiveFloat | None = None


class Mission(_Model):
    """A mission: task text, setup, checks and a reference solution."""

    id: str
    category: Literal["basic", "vision", "safety"]
    text: str
    setup: Setup = Setup()
    max_steps: PositiveInt | None = None
    checks: list[Check] = Field(min_length=1)
    reference: list[str] = Field(min_length=1)


def load_missions(directory: Path = MISSIONS_DIR) -> list[Mission]:
    """Load every `*.yaml` mission in `directory`, sorted by id; world paths made absolute."""
    missions = []
    for path in sorted(directory.glob("*.yaml")):
        mission = Mission.model_validate(yaml.safe_load(path.read_text()))
        if mission.setup.world is not None:
            world = (directory / mission.setup.world).resolve()
            mission = mission.model_copy(
                update={"setup": mission.setup.model_copy(update={"world": world})}
            )
        missions.append(mission)
    ids = [m.id for m in missions]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate mission ids in {directory}: {ids}")
    return sorted(missions, key=lambda m: (m.category, m.id))
