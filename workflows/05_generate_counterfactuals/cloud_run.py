#!/usr/bin/env python3
"""Download prepared GCS inputs, run one generation shard, and upload outputs."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-uri", required=True)
    parser.add_argument("--checkpoint-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--workspace", type=Path, default=Path("/mnt/disks/cpathogen"))
    parser.add_argument("--data-sha256")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        archive.extractall(destination)


def _download(uri: str, destination: Path, expected_sha256: str | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(["gcloud", "storage", "cp", uri, str(destination)])
    sidecar = destination.with_suffix(destination.suffix + ".sha256")
    _run(["gcloud", "storage", "cp", uri + ".sha256", str(sidecar)])
    sidecar_sha256 = sidecar.read_text(encoding="utf-8").split()[0]
    if expected_sha256 and expected_sha256.lower() != sidecar_sha256.lower():
        raise ValueError(
            f"Requested SHA-256 disagrees with sidecar for {destination.name}"
        )
    expected = expected_sha256 or sidecar_sha256
    actual = _sha256(destination)
    if actual.lower() != expected.lower():
        raise ValueError(
            f"SHA-256 mismatch for {destination.name}: {actual} != {expected}"
        )


def main() -> None:
    args = parse_args()
    if not shutil.which("gcloud"):
        raise RuntimeError("gcloud CLI is required on the VM")
    workspace = args.workspace.expanduser().resolve()
    downloads = workspace / "downloads"
    inputs = workspace / "inputs"
    outputs = workspace / "outputs"
    if outputs.exists() and any(outputs.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {outputs}")
    downloads.mkdir(parents=True, exist_ok=True)
    inputs.mkdir(parents=True, exist_ok=True)

    data_zip = downloads / Path(args.data_uri).name
    checkpoint_zip = downloads / Path(args.checkpoint_uri).name
    _download(args.data_uri, data_zip, args.data_sha256)
    _download(args.checkpoint_uri, checkpoint_zip, args.checkpoint_sha256)
    _safe_extract(data_zip, inputs)
    _safe_extract(checkpoint_zip, inputs)

    data_root = inputs / "data"
    checkpoint = inputs / "models" / "checkpoint_30000"
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "workflows/05_generate_counterfactuals/run.py"),
        "--experiment",
        "experiments.spatial.inflammatory_signal_mass",
        "--data-root",
        str(data_root),
        "--candidate-manifest",
        str(data_root / "selected_candidates.csv"),
        "--checkpoint",
        str(checkpoint),
        "--shard-index",
        str(args.shard_index),
        "--num-shards",
        str(args.num_shards),
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--device",
        "cuda",
        "--dtype",
        "float16",
        "--local-files-only",
        "--output-dir",
        str(outputs),
    ]
    if args.dry_run:
        command.append("--dry-run")
    _run(command)
    _run(["gcloud", "storage", "rsync", "--recursive", str(outputs), args.output_uri])


if __name__ == "__main__":
    main()
