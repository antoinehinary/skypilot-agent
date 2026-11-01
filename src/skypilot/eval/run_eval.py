"""Run the mission suite headless and score each run against ground truth.

Usage:
    python -m skypilot.eval.run_eval                         # qwen3:8b, config/agent.yaml
    python -m skypilot.eval.run_eval --think --label "qwen3:8b think"
    python -m skypilot.eval.run_eval --planning-interval 3 --label "qwen3:8b plan"
    python -m skypilot.eval.run_eval --model ollama_chat/qwen2.5vl:7b
    python -m skypilot.eval.run_eval --reference             # scripted reference solutions

Each run writes results/<timestamp>_<label>/: results.csv, summary.md, and per mission a
safety log and a JSON transcript.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from smolagents import LogLevel, Model

from skypilot.agent.run_agent import ServerOptions, build_model, count_action_steps, run_mission
from skypilot.config import CONFIG_DIR, load_agent_config, load_world_config
from skypilot.eval.checks import (
    Outcome,
    evaluate,
    read_decisions,
    repeated_refusals,
    report_position_error,
)
from skypilot.eval.missions import MISSIONS_DIR, Mission, load_missions
from skypilot.eval.scripted import ScriptedModel
from skypilot.mcp_server.server import parse_state

logger = logging.getLogger(__name__)
RESULTS_DIR = CONFIG_DIR.parent / "results"


@dataclass
class MissionRecord:
    """One scored mission run (a row of results.csv)."""

    label: str
    model_id: str
    mission: str
    category: str
    success: bool
    failed_checks: str
    steps: int
    input_tokens: int
    output_tokens: int
    sim_time_s: float
    wall_time_s: float
    refusals: int
    repeated_refusals: int
    report_position_error_m: float | None
    run_state: str
    error: str = ""
    check_details: list[str] = field(default_factory=list)


def run_one(
    mission: Mission,
    model: Model,
    *,
    label: str,
    model_id: str,
    max_steps: int,
    planning_interval: int | None,
    out_dir: Path,
) -> MissionRecord:
    """Fly one mission, score it, and save its safety log and transcript under `out_dir`."""
    log = out_dir / "logs" / f"{mission.id}.safety.jsonl"
    log.unlink(missing_ok=True)
    server = ServerOptions(
        world=mission.setup.world,
        initial_battery_pct=mission.setup.initial_battery_pct,
        safety_log=log,
        log_level="ERROR",  # refusals are in the safety log; keep the progress output clean
    )
    start = time.monotonic()
    base = {
        "label": label,
        "model_id": model_id,
        "mission": mission.id,
        "category": mission.category,
    }
    try:
        result = run_mission(
            mission.text,
            model,
            max_steps=mission.max_steps or max_steps,
            server=server,
            verbosity=LogLevel.OFF,
            planning_interval=planning_interval,
        )
    except Exception as e:  # noqa: BLE001 - one crashed mission must not stop the batch; it is recorded
        (out_dir / "logs" / f"{mission.id}.error.txt").write_text(traceback.format_exc())
        return MissionRecord(
            **base,
            success=False,
            failed_checks="crashed",
            steps=0,
            input_tokens=0,
            output_tokens=0,
            sim_time_s=0.0,
            wall_time_s=time.monotonic() - start,
            refusals=0,
            repeated_refusals=0,
            report_position_error_m=None,
            run_state="crashed",
            error=f"{type(e).__name__}: {e}",
        )
    wall = time.monotonic() - start
    run = result.run
    report = str(run.output)
    final = parse_state(result.final_state)
    decisions = read_decisions(log)
    outcome = Outcome(final, decisions, report, load_world_config(mission.setup.world))
    checks = [evaluate(check, outcome) for check in mission.checks]
    transcript = {
        "mission": mission.model_dump(mode="json"),
        "report": report,
        "final_state": result.final_state,
        "steps": run.steps,
    }
    (out_dir / "logs" / f"{mission.id}.transcript.json").write_text(
        json.dumps(transcript, indent=1, default=str)
    )
    usage = run.token_usage
    failed = [c.check for c in checks if not c.passed]
    if run.state != "success":  # ran out of steps: the agent never completed the mission
        failed.append("unfinished")
    return MissionRecord(
        **base,
        success=not failed,
        failed_checks=";".join(failed),
        steps=count_action_steps(run),
        input_tokens=usage.input_tokens if usage else 0,
        output_tokens=usage.output_tokens if usage else 0,
        sim_time_s=final.sim_time_s,
        wall_time_s=round(wall, 1),
        refusals=sum(not d["allowed"] for d in decisions),
        repeated_refusals=repeated_refusals(decisions),
        report_position_error_m=report_position_error(report, final),
        run_state=run.state,
        check_details=[f"{'PASS' if c.passed else 'FAIL'} {c.check}: {c.detail}" for c in checks],
    )


def write_results(records: list[MissionRecord], out_dir: Path) -> None:
    """results.csv (one row per mission) and a short summary.md."""
    columns = [f for f in MissionRecord.__dataclass_fields__ if f != "check_details"]
    with (out_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in records:
            writer.writerow({k: v for k, v in asdict(r).items() if k in columns})
    lines = [
        f"# {records[0].label}\n",
        "| Mission | Result | Steps | Wall time | Details |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in records:
        details = r.error or "<br>".join(r.check_details)
        lines.append(
            f"| {r.mission} | {'PASS' if r.success else 'FAIL'} | {r.steps} | "
            f"{r.wall_time_s:.0f} s | {details} |"
        )
    passed = sum(r.success for r in records)
    lines.append(f"\n**{passed}/{len(records)} missions passed.**")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run the selected missions and write results; 0 if all pass."""
    parser = argparse.ArgumentParser(description="Run the mission suite and score it.")
    parser.add_argument("--model", help="LiteLLM model id (default: config/agent.yaml)")
    parser.add_argument("--think", action="store_true", help="enable Ollama thinking")
    parser.add_argument("--planning-interval", type=int, help="forced plan every N steps")
    parser.add_argument("--label", help="name of this configuration in the results")
    parser.add_argument("--reference", action="store_true", help="replay reference solutions")
    parser.add_argument("--only", help="regex on mission id or category")
    parser.add_argument("--missions-dir", type=Path, default=MISSIONS_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    config = load_agent_config()
    if args.model:
        config = config.model_copy(update={"model_id": args.model})
    if args.think:
        config = config.model_copy(
            update={"ollama": config.ollama.model_copy(update={"think": True})}
        )
    planning = args.planning_interval or config.planning_interval
    model_id = "reference" if args.reference else config.model_id
    label = args.label or (
        model_id.split("/")[-1]
        + (" think" if args.think else "")
        + (f" plan{planning}" if planning else "")
    )

    missions = [
        m
        for m in load_missions(args.missions_dir)
        if not args.only or re.search(args.only, f"{m.id} {m.category}")
    ]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = RESULTS_DIR / f"{stamp}_{re.sub(r'[^A-Za-z0-9.]+', '-', label)}"
    (out_dir / "logs").mkdir(parents=True)
    print(f"{label}: {len(missions)} missions -> {out_dir}")

    records = []
    for mission in missions:
        model = (
            ScriptedModel.from_code(mission.reference) if args.reference else build_model(config)
        )
        record = run_one(
            mission,
            model,
            label=label,
            model_id=model_id,
            max_steps=config.max_steps,
            planning_interval=planning,
            out_dir=out_dir,
        )
        records.append(record)
        status = "PASS" if record.success else f"FAIL ({record.failed_checks or record.run_state})"
        print(
            f"  {mission.id:<24} {status:<40} {record.steps:>2} steps {record.wall_time_s:>6.1f} s"
        )
        write_results(records, out_dir)  # keep partial results if the batch is interrupted
    passed = sum(r.success for r in records)
    print(f"{passed}/{len(records)} passed")
    return 0 if passed == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
