#!/usr/bin/env python
"""Run one or all PathOGen fidelity experiments from the Colab asset config."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.colab.layout import DEFAULT_CONFIG, RuntimePaths

EXPERIMENT_ORDER = ("morphology", "spatial-count", "spatial-coordinate")
SCRIPT_NAMES = {
    "morphology": "02_morphology_fidelity.py",
    "spatial-count": "03_spatial_count_fidelity.py",
    "spatial-coordinate": "04_spatial_coordinate_fidelity.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=(*EXPERIMENT_ORDER, "all"),
        default=["all"],
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--cellvit-root", type=Path)
    parser.add_argument("--cellvit-model", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--num-images", type=int, default=25)
    parser.add_argument("--stems", nargs="*")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--spatial-strength", type=float, default=1.0)
    parser.add_argument(
        "--generator-precision", choices=("auto", "fp16", "fp32"), default="auto"
    )
    parser.add_argument(
        "--cellvit-precision", choices=("auto", "fp16", "fp32"), default="auto"
    )
    parser.add_argument("--features", nargs="+")
    parser.add_argument("--quantile-shift", type=float, default=0.20)
    parser.add_argument("--range-lower-quantile", type=float, default=0.01)
    parser.add_argument("--range-upper-quantile", type=float, default=0.99)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--max-match-distance", type=float, default=32.0)
    parser.add_argument("--guidance-hook")
    parser.add_argument("--guidance-config", type=Path)
    parser.add_argument("--max-guidance-attempts", type=int, default=1)
    parser.add_argument("--keep-rejected", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one two-step spatial-count case through generation and CellViT++",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print and save commands without executing them",
    )
    return parser.parse_args()


def selected_experiments(values: list[str]) -> list[str]:
    if "all" in values:
        return list(EXPERIMENT_ORDER)
    requested = set(values)
    return [name for name in EXPERIMENT_ORDER if name in requested]


def choose(override: Path | None, configured: Path | None) -> Path | None:
    return override.expanduser().resolve() if override is not None else configured


def append_flag(command: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        command.append(flag)


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Runtime config missing: {config_path}. Run experiments/colab/setup_colab.py first."
        )
    paths = RuntimePaths.read(config_path)
    data_dir = choose(args.data_dir, paths.data_dir)
    checkpoint_dir = choose(args.checkpoint_dir, paths.checkpoint_dir)
    cellvit_root = choose(args.cellvit_root, paths.cellvit_root)
    cellvit_model = choose(args.cellvit_model, paths.cellvit_model)
    output_root = choose(args.output_root, paths.output_root)
    assert data_dir is not None and output_root is not None

    experiments = selected_experiments(args.experiments)
    num_images = args.num_images
    steps = args.steps
    bootstrap = args.bootstrap
    if args.smoke_test:
        experiments = ["spatial-count"]
        num_images = 1
        steps = 2
        bootstrap = 0
    if not args.dry_run and not args.print_only:
        missing = [
            name
            for name, value in (
                ("checkpoint_dir", checkpoint_dir),
                ("cellvit_root", cellvit_root),
                ("cellvit_model", cellvit_model),
            )
            if value is None or not value.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing runtime assets: {missing}. Rerun setup_colab.py with the required paths."
            )

    output_root.mkdir(parents=True, exist_ok=True)
    scripts_root = paths.repo_root / "experiments"
    commands: list[list[str]] = []
    for experiment in experiments:
        output_dir = (
            output_root / "morphology"
            if experiment == "morphology"
            else output_root / "spatial"
        )
        command = [
            sys.executable,
            str(scripts_root / SCRIPT_NAMES[experiment]),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--num-images",
            str(num_images),
            "--steps",
            str(steps),
            "--seed",
            str(args.seed),
            "--bootstrap",
            str(bootstrap),
            "--spatial-strength",
            str(args.spatial_strength),
            "--generator-precision",
            args.generator_precision,
            "--cellvit-precision",
            args.cellvit_precision,
        ]
        if checkpoint_dir is not None:
            command.extend(["--checkpoint-dir", str(checkpoint_dir)])
        if cellvit_root is not None:
            command.extend(["--cellvit-root", str(cellvit_root)])
        if cellvit_model is not None:
            command.extend(["--cellvit-model", str(cellvit_model)])
        if args.stems:
            command.extend(["--stems", *args.stems])
        if args.guidance_hook:
            command.extend(["--guidance-hook", args.guidance_hook])
        if args.guidance_config:
            command.extend(["--guidance-config", str(args.guidance_config.resolve())])
        command.extend(["--max-guidance-attempts", str(args.max_guidance_attempts)])
        append_flag(command, args.keep_rejected, "--keep-rejected")
        append_flag(command, args.dry_run, "--dry-run")
        append_flag(command, args.analysis_only, "--analysis-only")
        append_flag(command, args.overwrite, "--overwrite")

        if experiment == "morphology":
            command.extend(
                [
                    "--quantile-shift",
                    str(args.quantile_shift),
                    "--range-lower-quantile",
                    str(args.range_lower_quantile),
                    "--range-upper-quantile",
                    str(args.range_upper_quantile),
                ]
            )
            if args.features:
                command.extend(["--features", *args.features])
        elif experiment == "spatial-coordinate":
            command.extend(
                [
                    "--grid-size",
                    str(args.grid_size),
                    "--max-match-distance",
                    str(args.max_match_distance),
                ]
            )
        commands.append(command)

    manifest = {
        "config": str(config_path),
        "experiments": experiments,
        "smoke_test": args.smoke_test,
        "commands": commands,
    }
    manifest_path = output_root / "suite_commands.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[suite] Command manifest: {manifest_path}")
    if experiments == list(EXPERIMENT_ORDER) and not args.dry_run:
        total_generations = num_images * 10
        print(
            f"[suite] Full default suite requests approximately {total_generations} generations "
            f"({num_images} x 9 morphology conditions plus {num_images} shared spatial baselines)."
        )
    for index, command in enumerate(commands, start=1):
        print(f"[suite {index}/{len(commands)}] {shlex.join(command)}", flush=True)
        if not args.print_only:
            subprocess.run(command, check=True, cwd=paths.repo_root)
    print(f"[suite] Completed: {', '.join(experiments)}")


if __name__ == "__main__":
    main()
