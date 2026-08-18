#!/usr/bin/env python3
"""Organize selected public GCS counterfactual panels by experiment and tile."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

SOURCE_BUCKET = "cpathogen_artifacts"
DEFAULT_GCS_DESTINATION = (
    "gs://cpathogen_artifacts/organized_counterfactuals"
)
EXPECTED_CONDITIONS = {
    "nuclear_enlargement": {
        "baseline",
        "nuclear_enlargement_minus_2p0sd",
        "nuclear_enlargement_minus_1p0sd",
        "nuclear_enlargement_plus_0p5sd",
        "nuclear_enlargement_plus_1p0sd",
        "nuclear_enlargement_plus_1p5sd",
        "nuclear_enlargement_plus_2p0sd",
    },
    "nuclear_shape_irregularity": {
        "baseline",
        "nuclear_shape_irregularity_minus_2p0sd",
        "nuclear_shape_irregularity_minus_1p0sd",
        "nuclear_shape_irregularity_plus_0p5sd",
        "nuclear_shape_irregularity_plus_1p0sd",
        "nuclear_shape_irregularity_plus_1p5sd",
        "nuclear_shape_irregularity_plus_2p0sd",
    },
    "stain_brightness": {
        "baseline",
        "stain_brightness_minus_2p0sd",
        "stain_brightness_minus_1p0sd",
        "stain_brightness_plus_0p5sd",
        "stain_brightness_plus_1p0sd",
        "stain_brightness_plus_1p5sd",
        "stain_brightness_plus_2p0sd",
    },
}
BASE_CONDITIONS = {
    "nuclear_enlargement": {
        "baseline",
        "nuclear_enlargement_plus_0p5sd",
        "nuclear_enlargement_plus_1p0sd",
        "nuclear_enlargement_plus_1p5sd",
    },
    "stain_brightness": {
        "baseline",
        "stain_brightness_plus_0p5sd",
        "stain_brightness_plus_1p0sd",
        "stain_brightness_plus_1p5sd",
    },
}
EXTENSION_CONDITIONS = {
    experiment: EXPECTED_CONDITIONS[experiment] - conditions
    for experiment, conditions in BASE_CONDITIONS.items()
}
EXPECTED_PANEL_COMPOSITION = {
    "nuclear_enlargement": {
        frozenset(EXPECTED_CONDITIONS["nuclear_enlargement"]): 770,
        frozenset(BASE_CONDITIONS["nuclear_enlargement"]): 230,
        frozenset(EXTENSION_CONDITIONS["nuclear_enlargement"]): 230,
    },
    "nuclear_shape_irregularity": {
        frozenset(EXPECTED_CONDITIONS["nuclear_shape_irregularity"]): 1000,
    },
    "stain_brightness": {
        frozenset(EXPECTED_CONDITIONS["stain_brightness"]): 770,
        frozenset(BASE_CONDITIONS["stain_brightness"]): 230,
        frozenset(EXTENSION_CONDITIONS["stain_brightness"]): 230,
    },
}


@dataclass(frozen=True)
class ManifestSource:
    experiment: str
    run_prefix: str
    manifest_name: str = "images.csv"

    @property
    def manifest_object(self) -> str:
        return f"{self.run_prefix.rstrip('/')}/{self.manifest_name}"


SOURCES = (
    ManifestSource(
        "nuclear_enlargement",
        "outputs/counterfactuals/nuclear_enlargement/sd_v1/"
        "cohort_1000/20260816-0230",
    ),
    ManifestSource(
        "nuclear_enlargement",
        "outputs/counterfactuals/nuclear_enlargement/sd_v1/"
        "cohort_1000/20260816-0230/extensions/signed_extremes_v2",
    ),
    ManifestSource(
        "nuclear_shape_irregularity",
        "outputs/counterfactuals/nuclear_shape_irregularity/sd_v2/"
        "cohort_1000/full_signed_panel",
    ),
    ManifestSource(
        "stain_brightness",
        "outputs/counterfactuals/stain_brightness/sd_v1/"
        "cohort_1000/20260816-053143",
    ),
    ManifestSource(
        "stain_brightness",
        "outputs/counterfactuals/stain_brightness/sd_v1/"
        "cohort_1000/20260816-053143/extensions/signed_extremes_v2",
    ),
)


@dataclass(frozen=True)
class OrganizedImage:
    experiment: str
    candidate_id: str
    stem: str
    seed: int
    condition: str
    source_object: str
    relative_destination: str

    @property
    def source_uri(self) -> str:
        return f"gs://{SOURCE_BUCKET}/{self.source_object}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy three completed CPathOGen experiments into an auditable "
            "experiment/tile/variation layout on Drive and, optionally, GCS, "
            "while preserving partial-panel provenance."
        )
    )
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument(
        "--gcs-destination",
        default=DEFAULT_GCS_DESTINATION,
        help="Non-destructive destination prefix, or an empty string to skip GCS.",
    )
    parser.add_argument(
        "--project",
        help="Google Cloud project used by authenticated destination writes.",
    )
    parser.add_argument(
        "--drive-workers",
        type=int,
        default=2,
        help="Conservative concurrency for Google Drive's mounted filesystem.",
    )
    parser.add_argument("--gcs-workers", type=int, default=8)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.drive_workers < 1 or args.gcs_workers < 1:
        parser.error("Worker counts must be positive")
    return args


def _safe_component(value: str, field: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError(f"Unsafe {field} path component: {value!r}")
    if value in {".", ".."}:
        raise ValueError(f"Unsafe {field} path component: {value!r}")
    return value


def _manifest_relative_image(image_path: str) -> PurePosixPath:
    parts = PurePosixPath(image_path.replace("\\", "/")).parts
    image_indices = [index for index, part in enumerate(parts) if part == "images"]
    if not image_indices:
        raise ValueError(f"Manifest image path has no images/ segment: {image_path}")
    relative = PurePosixPath(*parts[image_indices[-1] + 1 :])
    if len(relative.parts) != 3:
        raise ValueError(f"Unexpected manifest image layout: {image_path}")
    return relative


def plan_from_manifest_rows(
    source: ManifestSource,
    rows: Iterable[dict[str, str]],
) -> list[OrganizedImage]:
    planned: list[OrganizedImage] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            candidate_id = _safe_component(row["candidate_id"], "candidate_id")
            stem = _safe_component(row["stem"], "stem")
            condition = _safe_component(row["condition"], "condition")
            seed = int(row["seed"])
            relative_source = _manifest_relative_image(row["image_path"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid {source.manifest_object} row {row_number}"
            ) from error
        expected_filename = f"{condition}.png"
        if relative_source.name != expected_filename:
            raise ValueError(
                f"Condition/image mismatch in {source.manifest_object}: "
                f"{condition!r} versus {relative_source.name!r}"
            )
        source_object = (
            f"{source.run_prefix.rstrip('/')}/images/{relative_source.as_posix()}"
        )
        relative_destination = PurePosixPath(
            source.experiment, stem, expected_filename
        ).as_posix()
        planned.append(
            OrganizedImage(
                experiment=source.experiment,
                candidate_id=candidate_id,
                stem=stem,
                seed=seed,
                condition=condition,
                source_object=source_object,
                relative_destination=relative_destination,
            )
        )
    return planned


def validate_plan(items: list[OrganizedImage]) -> None:
    by_destination: dict[str, OrganizedImage] = {}
    by_experiment_and_stem: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in items:
        previous = by_destination.get(item.relative_destination)
        if previous is not None:
            raise ValueError(
                "Duplicate organized destination from source manifests: "
                f"{item.relative_destination}; {previous.source_uri}; "
                f"{item.source_uri}"
            )
        by_destination[item.relative_destination] = item
        by_experiment_and_stem[(item.experiment, item.stem)].add(item.condition)

    actual_experiments = {experiment for experiment, _ in by_experiment_and_stem}
    if actual_experiments != set(EXPECTED_PANEL_COMPOSITION):
        raise ValueError(f"Unexpected experiment set: {sorted(actual_experiments)}")

    actual_composition: dict[str, Counter[frozenset[str]]] = defaultdict(Counter)
    for (experiment, _), conditions in by_experiment_and_stem.items():
        actual_composition[experiment][frozenset(conditions)] += 1
    for experiment, expected in EXPECTED_PANEL_COMPOSITION.items():
        if actual_composition[experiment] != Counter(expected):
            printable = {
                "|".join(sorted(conditions)): count
                for conditions, count in actual_composition[experiment].items()
            }
            raise ValueError(
                f"Unexpected panel composition for {experiment}: {printable}"
            )

    expected_total = 1000 * sum(len(value) for value in EXPECTED_CONDITIONS.values())
    if len(items) != expected_total:
        raise ValueError(f"Planned {len(items)} images; expected {expected_total}")


def panel_summary(items: list[OrganizedImage]) -> dict[str, dict[str, int]]:
    by_panel: dict[tuple[str, str], int] = Counter(
        (item.experiment, item.stem) for item in items
    )
    summary: dict[str, dict[str, int]] = {}
    for experiment, expected_conditions in EXPECTED_CONDITIONS.items():
        panel_sizes = [
            count
            for (panel_experiment, _), count in by_panel.items()
            if panel_experiment == experiment
        ]
        expected_size = len(expected_conditions)
        summary[experiment] = {
            "images": sum(panel_sizes),
            "tile_folders": len(panel_sizes),
            "complete_panels": sum(count == expected_size for count in panel_sizes),
            "partial_panels": sum(count != expected_size for count in panel_sizes),
        }
    return summary


def _parse_gs_prefix(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI: {uri}")
    bucket, separator, prefix = uri[5:].partition("/")
    if not bucket or not separator or not prefix.strip("/"):
        raise ValueError(f"GCS destination must include a bucket and prefix: {uri}")
    return bucket, prefix.strip("/")


def _write_plan_manifest(path: Path, items: list[OrganizedImage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = [*asdict(items[0]).keys(), "source_uri"]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow({**asdict(item), "source_uri": item.source_uri})
    temporary.replace(path)


def _parallel_phase(
    label: str,
    items: list[OrganizedImage],
    worker: Any,
    *,
    workers: int,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker, item): item for item in items}
        for completed, future in enumerate(as_completed(futures), start=1):
            item = futures[future]
            try:
                counts[str(future.result())] += 1
            except Exception as error:
                raise RuntimeError(
                    f"{label} failed for {item.source_uri} -> "
                    f"{item.relative_destination}"
                ) from error
            if completed % 100 == 0 or completed == len(items):
                print(
                    f"[{label}] {completed}/{len(items)} "
                    f"written={counts['written']} skipped={counts['skipped']}",
                    flush=True,
                )
    return counts


def main() -> None:
    args = parse_args()
    try:
        from google.api_core.exceptions import PreconditionFailed
        from google.cloud import storage
    except ImportError as error:
        raise RuntimeError(
            "Install the cloud extra first: pip install -e '.[cloud]'"
        ) from error

    anonymous = storage.Client.create_anonymous_client()
    source_bucket = anonymous.bucket(SOURCE_BUCKET)
    planned: list[OrganizedImage] = []
    for source in SOURCES:
        payload = source_bucket.blob(source.manifest_object).download_as_bytes()
        rows = csv.DictReader(io.StringIO(payload.decode("utf-8")))
        source_items = plan_from_manifest_rows(source, rows)
        planned.extend(source_items)
        print(
            f"[manifest] {source.experiment}: +{len(source_items)} from "
            f"gs://{SOURCE_BUCKET}/{source.manifest_object}",
            flush=True,
        )
    validate_plan(planned)
    planned.sort(key=lambda item: item.relative_destination)

    drive_root = args.drive_root.expanduser().resolve()
    drive_root.mkdir(parents=True, exist_ok=True)
    plan_path = drive_root / "organized_bucket_images.csv"
    _write_plan_manifest(plan_path, planned)
    panels = panel_summary(planned)
    print(f"Validated {len(planned)} images across three experiments")
    print(json.dumps(panels, indent=2, sort_keys=True))
    if args.plan_only:
        return

    def download_to_drive(item: OrganizedImage) -> str:
        destination = drive_root / Path(item.relative_destination)
        if destination.is_file() and destination.stat().st_size > 0:
            return "skipped"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            source_bucket.blob(item.source_object).download_to_filename(temporary)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return "written"

    drive_counts = _parallel_phase(
        "Drive",
        planned,
        download_to_drive,
        workers=args.drive_workers,
    )

    gcs_counts: Counter[str] = Counter()
    destination_uri = args.gcs_destination.strip()
    destination_manifest_uri = None
    if destination_uri:
        destination_bucket_name, destination_prefix = _parse_gs_prefix(
            destination_uri
        )
        authenticated = storage.Client(project=args.project)
        authenticated_source = authenticated.bucket(SOURCE_BUCKET)
        destination_bucket = authenticated.bucket(destination_bucket_name)

        def copy_to_gcs(item: OrganizedImage) -> str:
            destination_name = (
                f"{destination_prefix}/{item.relative_destination}"
            )
            source_blob = authenticated_source.blob(item.source_object)
            try:
                authenticated_source.copy_blob(
                    source_blob,
                    destination_bucket,
                    destination_name,
                    if_generation_match=0,
                )
            except PreconditionFailed:
                return "skipped"
            return "written"

        gcs_counts = _parallel_phase(
            "GCS",
            planned,
            copy_to_gcs,
            workers=args.gcs_workers,
        )
        destination_manifest_name = (
            f"{destination_prefix}/organized_bucket_images.csv"
        )
        destination_bucket.blob(destination_manifest_name).upload_from_filename(
            plan_path
        )
        destination_manifest_uri = (
            f"gs://{destination_bucket_name}/{destination_manifest_name}"
        )

    summary = {
        "schema_version": 1,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "image_count": len(planned),
        "experiment_counts": dict(
            sorted(Counter(item.experiment for item in planned).items())
        ),
        "panel_summary": panels,
        "drive_root": str(drive_root),
        "drive_counts": dict(sorted(drive_counts.items())),
        "gcs_destination": destination_uri or None,
        "gcs_counts": dict(sorted(gcs_counts.items())),
        "gcs_manifest": destination_manifest_uri,
        "source_bucket_unchanged": True,
    }
    summary_path = drive_root / "organized_bucket_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
