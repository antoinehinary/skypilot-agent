"""Results tables and README update, on synthetic results.csv files."""

import csv
from pathlib import Path

import pytest

from skypilot.eval.report import (
    END,
    START,
    comparison_table,
    latest_runs,
    mission_table,
    update_readme,
)

FIELDS = [
    "label",
    "mission",
    "category",
    "success",
    "failed_checks",
    "steps",
    "input_tokens",
    "output_tokens",
    "wall_time_s",
    "repeated_refusals",
    "report_position_error_m",
    "run_state",
]


def _write_run(directory: Path, label: str, rows: list[dict]) -> None:
    directory.mkdir(parents=True)
    with (directory / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "label": label,
                    "failed_checks": "",
                    "run_state": "success",
                    "report_position_error_m": "",
                    **row,
                }
            )


def _row(mission, category, success, steps=4, tokens=(1000, 100), wall=10.0, retries=0, err=""):
    return {
        "mission": mission,
        "category": category,
        "success": str(success),
        "steps": steps,
        "input_tokens": tokens[0],
        "output_tokens": tokens[1],
        "wall_time_s": wall,
        "repeated_refusals": retries,
        "report_position_error_m": err,
        "failed_checks": "" if success else "final_position",
    }


@pytest.fixture
def results(tmp_path: Path) -> Path:
    _write_run(tmp_path / "20261101-100000_old", "model-a", [_row("goto", "basic", False)])
    _write_run(
        tmp_path / "20261101-110000_new",
        "model-a",
        [
            _row("goto", "basic", True, err="0.02"),
            _row("find_red", "vision", False, steps=8, retries=2, err="2.0"),
        ],
    )
    _write_run(tmp_path / "20261101-120000_ref", "reference", [_row("goto", "basic", True)])
    return tmp_path


def test_latest_run_per_label_wins_and_reference_is_excluded(results: Path) -> None:
    runs = latest_runs(results)

    assert list(runs) == ["model-a"]
    assert [r["mission"] for r in runs["model-a"]] == ["goto", "find_red"]


def test_comparison_table_row(results: Path) -> None:
    table = comparison_table(latest_runs(results))

    row = table.splitlines()[2]
    assert row == "| `model-a` | **1/2** | 1/1 | 0/1 | - | 6.0 | 1.1k | 10 s | 2 | 1/2 |"


def test_mission_table_shows_failed_check(results: Path) -> None:
    table = mission_table(latest_runs(results))

    assert "| find_red | fail (final_position) |" in table
    assert "| goto | pass |" in table


def test_update_readme_replaces_only_between_markers(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{START}\nold table\n{END}\noutro\n")

    update_readme("new table", readme)

    assert readme.read_text() == f"intro\n{START}\nnew table\n{END}\noutro\n"


def test_update_readme_requires_markers(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("no markers here\n")

    with pytest.raises(ValueError, match="markers"):
        update_readme("table", readme)


def test_a_pass_that_never_finished_counts_as_a_failure(tmp_path: Path) -> None:
    lucky = _row("geofence", "safety", True)
    lucky["run_state"] = "max_steps_error"
    _write_run(tmp_path / "20261101-130000_run", "model-b", [lucky])

    runs = latest_runs(tmp_path)

    assert "| `model-b` | **0/1** |" in comparison_table(runs)
    assert "| geofence | fail (unfinished) |" in mission_table(runs)
