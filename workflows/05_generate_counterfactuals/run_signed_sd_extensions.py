#!/usr/bin/env python3
"""Run the missing signed-SD generation jobs sequentially on one GPU VM."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

WORKFLOW_ROOT = Path(__file__).resolve().parent
DEFAULT_PLAN = WORKFLOW_ROOT / "signed_sd_extensions_v2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-uri",
        default=(
            "gs://cpathogen_artifacts/models/pathogen_phase2_checkpoint_30000.zip"
        ),
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--workspace", type=Path, default=Path("/mnt/disks/cpathogen-sd-extensions")
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sync-interval-seconds", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_jobs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError(f"No jobs declared in {path}")
    for job in jobs:
        required = {
            "name",
            "experiment",
            "data_uri",
            "output_uri",
            "candidate_count",
            "omit_baseline",
            "interventions",
        }
        missing = required.difference(job)
        if missing:
            raise ValueError(f"Job is missing {sorted(missing)}: {job}")
        expected_interventions = 3 if job["omit_baseline"] else 6
        if len(job["interventions"]) != expected_interventions:
            raise ValueError(
                f"Expected {expected_interventions} interventions: {job['name']}"
            )
    return jobs


def main() -> None:
    args = parse_args()
    jobs = load_jobs(args.plan)
    cloud_runner = WORKFLOW_ROOT / "cloud_run.py"
    outputs = args.workspace.expanduser().resolve() / "outputs"

    for index, job in enumerate(jobs, start=1):
        if outputs.exists() and any(outputs.iterdir()):
            if index == 1:
                raise FileExistsError(
                    f"Refusing to replace a non-empty initial output directory: {outputs}"
                )
            shutil.rmtree(outputs)
        command = [
            sys.executable,
            str(cloud_runner),
            "--data-uri",
            job["data_uri"],
            "--checkpoint-uri",
            args.checkpoint_uri,
            "--output-uri",
            job["output_uri"],
            "--experiment",
            job["experiment"],
            "--workspace",
            str(args.workspace),
            "--steps",
            str(args.steps),
            "--batch-size",
            str(args.batch_size),
            "--sync-interval-seconds",
            str(args.sync_interval_seconds),
        ]
        if job["omit_baseline"]:
            command.append("--omit-baseline")
        for intervention in job["interventions"]:
            command.extend(["--intervention", intervention])
        if args.dry_run:
            command.append("--dry-run")
        print(f"[{index}/{len(jobs)}] Starting {job['name']}", flush=True)
        subprocess.run(command, check=True)

    print(f"Completed all {len(jobs)} signed-SD extension jobs", flush=True)


if __name__ == "__main__":
    main()
