#!/usr/bin/env python3
"""Download prepared GCS inputs, run one generation shard, and upload outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    parser.add_argument("--sync-interval-seconds", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


class ProgressUploader:
    """Publish generation progress and completed files while inference runs."""

    def __init__(
        self,
        *,
        workspace: Path,
        outputs: Path,
        output_uri: str,
        interval_seconds: int,
    ) -> None:
        if interval_seconds < 15:
            raise ValueError("--sync-interval-seconds must be at least 15")
        self.workspace = workspace
        self.outputs = outputs
        self.output_uri = output_uri.rstrip("/")
        self.interval_seconds = interval_seconds
        self.status_path = workspace / "status.json"
        self._state: dict[str, Any] = {
            "phase": "starting",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "error": None,
        }
        self._lock = threading.Lock()
        self._status_write_lock = threading.Lock()
        self._sync_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def update(self, phase: str, **values: Any) -> None:
        with self._lock:
            self._state.update(values)
            self._state["phase"] = phase
        self._write_status()

    def start(self) -> None:
        self._write_status()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval_seconds + 30)

    def _counts(self) -> dict[str, int | None]:
        png_count = sum(1 for _ in self.outputs.glob("images/**/*.png"))
        pairs_path = self.outputs / "pairs.jsonl"
        pair_count = None
        if pairs_path.is_file():
            with pairs_path.open("r", encoding="utf-8") as handle:
                pair_count = sum(1 for _ in handle)
        expected_png_count = None
        manifest_path = self.outputs / "run_manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                candidates = int(manifest["data"]["candidate_count"])
                interventions = len(manifest["experiment"]["interventions"])
                expected_png_count = candidates * (interventions + 1)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
        return {
            "generated_png_count": png_count,
            "matched_pair_count": pair_count,
            "expected_png_count": expected_png_count,
        }

    def _write_status(self) -> None:
        with self._status_write_lock:
            with self._lock:
                payload = dict(self._state)
            payload.update(self._counts())
            payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            temporary = self.status_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.status_path)

    def sync(self) -> None:
        with self._sync_lock:
            self._sync()

    def _sync(self) -> None:
        self._write_status()
        if self.outputs.is_dir():
            subprocess.run(
                [
                    "gcloud",
                    "storage",
                    "rsync",
                    "--recursive",
                    str(self.outputs),
                    self.output_uri,
                ],
                check=True,
            )
        subprocess.run(
            [
                "gcloud",
                "storage",
                "cp",
                str(self.status_path),
                self.output_uri + "/status.json",
            ],
            check=True,
        )

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.sync()
            except (OSError, subprocess.CalledProcessError) as error:
                print(f"Progress upload warning: {error}", file=sys.stderr, flush=True)
            self._stop.wait(self.interval_seconds)


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
    uploader = ProgressUploader(
        workspace=workspace,
        outputs=outputs,
        output_uri=args.output_uri,
        interval_seconds=args.sync_interval_seconds,
    )
    uploader.start()
    try:
        uploader.update("downloading_inputs")
        data_zip = downloads / Path(args.data_uri).name
        checkpoint_zip = downloads / Path(args.checkpoint_uri).name
        _download(args.data_uri, data_zip, args.data_sha256)
        _download(args.checkpoint_uri, checkpoint_zip, args.checkpoint_sha256)

        uploader.update("extracting_inputs")
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
        uploader.update("generating")
        _run(command)
        uploader.update(
            "completed", completed_at=datetime.now(timezone.utc).isoformat()
        )
        uploader.sync()
    except Exception as error:
        uploader.update(
            "failed",
            failed_at=datetime.now(timezone.utc).isoformat(),
            error=f"{type(error).__name__}: {error}",
        )
        try:
            uploader.sync()
        except (OSError, subprocess.CalledProcessError) as sync_error:
            print(f"Final progress upload failed: {sync_error}", file=sys.stderr)
        raise
    finally:
        uploader.stop()


if __name__ == "__main__":
    main()
