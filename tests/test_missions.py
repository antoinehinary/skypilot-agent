"""The mission suite: every file is valid, and every reference solution passes its own checks."""

import pytest

from skypilot.config import load_world_config
from skypilot.eval.missions import FinalPosition, load_missions

MISSIONS = load_missions()


def test_suite_covers_every_category() -> None:
    assert {m.category for m in MISSIONS} == {"basic", "vision", "safety"}


@pytest.mark.parametrize("mission", MISSIONS, ids=lambda m: m.id)
def test_referenced_objects_exist_in_the_mission_world(mission) -> None:
    world = {o.name for o in load_world_config(mission.setup.world).objects}
    for check in mission.checks:
        if isinstance(check, FinalPosition) and check.object is not None:
            assert check.object in world


@pytest.mark.sim
@pytest.mark.parametrize("mission", MISSIONS, ids=lambda m: m.id)
def test_reference_solution_passes_all_checks(mission, tmp_path) -> None:
    pytest.importorskip("pybullet")
    from skypilot.eval.run_eval import run_one
    from skypilot.eval.scripted import ScriptedModel

    (tmp_path / "logs").mkdir()
    record = run_one(
        mission,
        ScriptedModel.from_code(mission.reference),
        label="reference",
        model_id="reference",
        max_steps=len(mission.reference) + 2,
        planning_interval=None,
        out_dir=tmp_path,
    )

    assert record.success, record.error or "\n".join(record.check_details)
