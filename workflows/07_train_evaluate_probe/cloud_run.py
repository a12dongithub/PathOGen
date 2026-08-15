#!/usr/bin/env python3
"""One-command GCS runner for CTransPath training and counterfactual scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAINING_URI = (
    "gs://cpathogen_artifacts/inputs/bcss_tumor_stroma_v1/bcss_tumor_stroma_v1.zip"
)
DEFAULT_CHECKPOINT_URI = "gs://cpathogen_artifacts/models/ctranspath/ctranspath.pth"
DEFAULT_COUNTERFACTUAL_URI = (
    "gs://cpathogen_artifacts/outputs/inflammatory_centroid_density_sd_v1_20260815-1508"
)


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data-uri", default=DEFAULT_TRAINING_URI)
    parser.add_argument("--encoder-checkpoint-uri", default=DEFAULT_CHECKPOINT_URI)
    parser.add_argument("--counterfactual-uri", default=DEFAULT_COUNTERFACTUAL_URI)
    parser.add_argument(
        "--output-uri",
        default=(
            "gs://cpathogen_artifacts/outputs/model_probes/"
            f"ctranspath_bcss_tumor_stroma_{timestamp}"
        ),
    )
    parser.add_argument(
        "--workspace", type=Path, default=Path.home() / "cpathogen-probe"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cuda", "cpu", "mps")
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive_path: Path, destination: Path) -> None:
    resolved = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != resolved and resolved not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        archive.extractall(destination)


def download_verified(uri: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar = destination.with_suffix(destination.suffix + ".sha256")
    run(["gcloud", "storage", "cp", uri + ".sha256", str(sidecar)])
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if destination.is_file() and sha256(destination).lower() == expected.lower():
        print(f"Reusing verified download: {destination}", flush=True)
        return
    run(["gcloud", "storage", "cp", uri, str(destination)])
    actual = sha256(destination)
    if actual.lower() != expected.lower():
        raise ValueError(f"SHA-256 mismatch for {destination}: {actual} != {expected}")


def write_status(path: Path, phase: str, **values: object) -> None:
    payload = {
        "phase": phase,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **values,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def upload_outputs(outputs: Path, output_uri: str, status_path: Path) -> None:
    if outputs.is_dir():
        run(["gcloud", "storage", "rsync", "--recursive", str(outputs), output_uri])
    run(
        [
            "gcloud",
            "storage",
            "cp",
            str(status_path),
            output_uri.rstrip("/") + "/status.json",
        ]
    )


def main() -> None:
    args = parse_args()
    if not shutil.which("gcloud"):
        raise RuntimeError("gcloud CLI is required on the VM")
    workspace = args.workspace.expanduser().resolve()
    downloads = workspace / "downloads"
    inputs = workspace / "inputs"
    outputs = workspace / "outputs"
    status_path = workspace / "status.json"
    if outputs.exists() and any(outputs.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {outputs}")
    downloads.mkdir(parents=True, exist_ok=True)
    inputs.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    write_status(status_path, "starting", output_uri=args.output_uri)
    try:
        write_status(
            status_path, "downloading_training_data", output_uri=args.output_uri
        )
        training_zip = downloads / Path(args.training_data_uri).name
        encoder_checkpoint = downloads / Path(args.encoder_checkpoint_uri).name
        download_verified(args.training_data_uri, training_zip)
        download_verified(args.encoder_checkpoint_uri, encoder_checkpoint)

        training_root = inputs / "bcss_tumor_stroma_v1"
        if not (training_root / "tiles.csv").is_file():
            write_status(
                status_path, "extracting_training_data", output_uri=args.output_uri
            )
            safe_extract(training_zip, inputs)

        counterfactual_root = inputs / "counterfactuals"
        counterfactual_root.mkdir(parents=True, exist_ok=True)
        write_status(
            status_path, "downloading_counterfactuals", output_uri=args.output_uri
        )
        run(
            [
                "gcloud",
                "storage",
                "rsync",
                "--recursive",
                args.counterfactual_uri.rstrip("/"),
                str(counterfactual_root),
            ]
        )
        image_count = sum(1 for _ in counterfactual_root.glob("images/**/*.png"))
        if image_count != 1_200:
            raise ValueError(f"Expected 1,200 counterfactual PNGs, found {image_count}")
        if not (counterfactual_root / "images.csv").is_file():
            raise FileNotFoundError(counterfactual_root / "images.csv")

        if args.dry_run:
            write_status(
                status_path,
                "dry_run_completed",
                output_uri=args.output_uri,
                counterfactual_png_count=image_count,
            )
            upload_outputs(outputs, args.output_uri, status_path)
            return

        write_status(
            status_path,
            "training_and_evaluating",
            output_uri=args.output_uri,
            counterfactual_png_count=image_count,
        )
        command = [
            sys.executable,
            str(REPOSITORY_ROOT / "workflows/07_train_evaluate_probe/run.py"),
            "--training-root",
            str(training_root),
            "--counterfactual-root",
            str(counterfactual_root),
            "--counterfactual-source-uri",
            args.counterfactual_uri,
            "--ctranspath-checkpoint",
            str(encoder_checkpoint),
            "--output-dir",
            str(outputs),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--device",
            args.device,
        ]
        run(command)
        write_status(
            status_path,
            "uploading_outputs",
            output_uri=args.output_uri,
            counterfactual_png_count=image_count,
        )
        upload_outputs(outputs, args.output_uri, status_path)
        write_status(
            status_path,
            "completed",
            output_uri=args.output_uri,
            counterfactual_png_count=image_count,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        run(
            [
                "gcloud",
                "storage",
                "cp",
                str(status_path),
                args.output_uri.rstrip("/") + "/status.json",
            ]
        )
    except Exception as error:
        write_status(
            status_path,
            "failed",
            output_uri=args.output_uri,
            error=f"{type(error).__name__}: {error}",
        )
        try:
            upload_outputs(outputs, args.output_uri, status_path)
        except (OSError, subprocess.CalledProcessError) as upload_error:
            print(f"Failure-status upload also failed: {upload_error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
