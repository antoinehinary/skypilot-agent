"""Combine evaluation runs into markdown tables and update the README between markers.

Usage:
    python -m skypilot.eval.report                  # print tables for the latest run per label
    python -m skypilot.eval.report --update-readme  # also rewrite the README results section
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

from skypilot.eval.checks import REPORT_POSITION_TOLERANCE_M
from skypilot.eval.run_eval import RESULTS_DIR

README = RESULTS_DIR.parent / "README.md"
START, END = "<!-- results:start -->", "<!-- results:end -->"
CATEGORIES = ("basic", "vision", "safety")

Row = dict[str, str]


def latest_runs(
    results_dir: Path = RESULTS_DIR, include_reference: bool = False
) -> dict[str, list[Row]]:
    """Rows of the most recent complete run for each label (directories sort by timestamp)."""
    runs: dict[str, list[Row]] = {}
    for run_dir in sorted(p for p in results_dir.iterdir() if (p / "results.csv").exists()):
        with (run_dir / "results.csv").open() as f:
            rows = list(csv.DictReader(f))
        if rows and (include_reference or rows[0]["label"] != "reference"):
            runs[rows[0]["label"]] = rows  # later directories overwrite earlier ones
    return runs


def _passed(rows: list[Row]) -> str:
    return f"{sum(r['success'] == 'True' for r in rows)}/{len(rows)}"


def comparison_table(runs: dict[str, list[Row]]) -> str:
    """One row per configuration: success overall and per category, cost, and honesty metrics."""
    header = (
        "| Configuration | Success | Basic | Vision | Safety | Steps / mission | "
        "Tokens / mission | Time / mission | Retried refusals | Reports with correct position |"
    )
    lines = [header, "| " + " | ".join(["---"] * 10) + " |"]
    for label, rows in runs.items():
        by_category = defaultdict(list)
        for r in rows:
            by_category[r["category"]].append(r)
        quoted = [float(r["report_position_error_m"]) for r in rows if r["report_position_error_m"]]
        accurate = sum(e <= REPORT_POSITION_TOLERANCE_M for e in quoted)
        tokens = mean(int(r["input_tokens"]) + int(r["output_tokens"]) for r in rows)
        cells = [
            f"`{label}`",
            f"**{_passed(rows)}**",
            *(_passed(by_category[c]) if by_category[c] else "-" for c in CATEGORIES),
            f"{mean(int(r['steps']) for r in rows):.1f}",
            f"{tokens / 1000:.1f}k",
            f"{mean(float(r['wall_time_s']) for r in rows):.0f} s",
            str(sum(int(r["repeated_refusals"]) for r in rows)),
            f"{accurate}/{len(quoted)}" if quoted else "-",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def mission_table(runs: dict[str, list[Row]]) -> str:
    """Pass/fail grid: one row per mission, one column per configuration."""
    labels = list(runs)
    missions: dict[str, dict[str, Row]] = defaultdict(dict)
    order: list[str] = []
    for label, rows in runs.items():
        for r in rows:
            if r["mission"] not in order:
                order.append(r["mission"])
            missions[r["mission"]][label] = r
    lines = [
        "| Mission | " + " | ".join(f"`{x}`" for x in labels) + " |",
        "| --- |" + " --- |" * len(labels),
    ]
    for mission in order:
        cells = []
        for label in labels:
            r = missions[mission].get(label)
            if r is None:
                cells.append("-")
            elif r["success"] == "True":
                cells.append("pass")
            else:
                cells.append(f"fail ({r['failed_checks'] or r['run_state']})")
        lines.append(f"| {mission} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def results_markdown(runs: dict[str, list[Row]]) -> str:
    n = max(len(rows) for rows in runs.values())
    tolerance = REPORT_POSITION_TOLERANCE_M
    details = (
        f"<details><summary>Per-mission results</summary>\n\n{mission_table(runs)}\n\n</details>"
    )
    return (
        f"{comparison_table(runs)}\n\n{details}\n\n"
        f"{n} missions, one run per configuration at temperature 0 with a fixed seed: repeatable "
        "within a session, but a rerun on another day can differ on some missions, so treat "
        "single-mission differences between configurations as indicative, not significant. "
        "Runs were "
        "headless on an Apple Silicon MacBook Pro, all models local in Ollama. "
        "Success is scored on the drone's verified final state and the safety log, never on the "
        f"agent's report; a report's position counts as correct within {tolerance} m."
    )


def update_readme(markdown: str, readme: Path = README) -> None:
    """Replace the text between the results markers in `readme`."""
    text = readme.read_text()
    if START not in text or END not in text:
        raise ValueError(f"{readme} has no {START} ... {END} markers")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    readme.write_text(f"{before}{START}\n{markdown}\n{END}{after}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise evaluation runs.")
    parser.add_argument("--update-readme", action="store_true")
    parser.add_argument("--include-reference", action="store_true")
    args = parser.parse_args(argv)
    runs = latest_runs(include_reference=args.include_reference)
    if not runs:
        print(f"no results in {RESULTS_DIR}")
        return 1
    markdown = results_markdown(runs)
    print(markdown)
    if args.update_readme:
        update_readme(markdown)
        print(f"\nupdated {README}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
