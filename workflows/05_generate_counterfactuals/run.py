#!/usr/bin/env python3
"""Generate matched-noise baseline/counterfactual images from Python plugins."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from cpathogen.counterfactuals import (
    CandidateRecord,
    ConditionBundle,
    ConditionStore,
    InterventionContext,
    MatchedPairRecord,
    load_candidate_manifest,
    load_interventions,
    select_candidate_shard,
    select_interventions,
)

DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data"
DEFAULT_IMAGES_DIR = REPOSITORY_ROOT / "data/images"
DEFAULT_CHECKPOINT = REPOSITORY_ROOT / "models/pathogen_phase2/checkpoint_30000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply Python-defined interventions in memory and generate matched-noise "
            "Phase-2 baseline/counterfactual images."
        )
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Import path or .py file defining build_interventions().",
    )
    parser.add_argument(
        "--intervention",
        action="append",
        dest="interventions",
        help="Run only this intervention slug; repeat for more than one.",
    )
    parser.add_argument("--list-interventions", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--spatial-maps-dir", type=Path)
    parser.add_argument("--morphology-table", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-model")
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--stem", action="append", dest="stems")
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        help="CSV with one candidate_id, stem, and selected seed per generation job.",
    )
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--num-tiles", type=int, default=1)
    parser.add_argument(
        "--all-tiles",
        action="store_true",
        help="Generate for every stem with both an aligned spatial map and morphology row.",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--intervention-seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--spatial-strength", type=float, default=2.0)
    parser.add_argument("--prompt", default="he")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Maximum number of conditions denoised together.",
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "mps", "cuda")
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float16", "bfloat16", "float32"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and apply interventions without loading the diffusion model.",
    )
    parser.add_argument(
        "--omit-baseline",
        action="store_true",
        help=(
            "Generate only selected interventions when extending a dataset whose "
            "matched baseline images already exist."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue an interrupted compatible output directory. Completed PNGs "
            "and pair records are verified and skipped."
        ),
    )
    args = parser.parse_args()
    if args.num_tiles < 1:
        parser.error("--num-tiles must be at least 1")
    if args.all_tiles and args.stems:
        parser.error("--all-tiles cannot be combined with --stem")
    if args.candidate_manifest and (args.all_tiles or args.stems or args.seeds):
        parser.error(
            "--candidate-manifest cannot be combined with --all-tiles, --stem, or --seed"
        )
    if (args.shard_index is None) != (args.num_shards is None):
        parser.error("--shard-index and --num-shards must be provided together")
    if args.shard_index is not None and not args.candidate_manifest:
        parser.error("Sharding requires --candidate-manifest")
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.spatial_strength < 0:
        parser.error("--spatial-strength must be non-negative")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.resume and args.dry_run:
        parser.error("--resume cannot be combined with --dry-run")
    return args


def _json_write(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _append_image_manifest(path: Path, record: dict[str, Any]) -> None:
    fieldnames = [
        "candidate_id",
        "stem",
        "seed",
        "condition",
        "intervention_parameters",
        "image_path",
    ]
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def _save_png(image: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    image.save(temporary, format="PNG")
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _completed_image_keys(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    completed: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_path = Path(row["image_path"])
            if image_path.is_file():
                completed.add((row["candidate_id"], row["condition"]))
    return completed


def _completed_pair_keys(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    completed: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = (record["candidate_id"], record["intervention"]["slug"])
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid pair record at {path}:{line_number}"
                ) from error
            completed.add(key)
    return completed


@dataclass
class _GenerationJob:
    candidate: CandidateRecord
    condition_name: str
    condition: ConditionBundle
    output_path: Path
    baseline_path: Path | None
    intervention: Any | None = None
    applied: Any | None = None
    difference: dict[str, Any] | None = None
    reference_tile: Path | None = None
    pair_already_recorded: bool = False


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _versions() -> dict[str, str | None]:
    output: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in ("cpathogen", "torch", "diffusers", "transformers", "pandas"):
        try:
            output[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            output[package] = None
    return output


def _select_stems(args: argparse.Namespace, store: ConditionStore) -> list[str]:
    if args.stems:
        missing = sorted(set(args.stems).difference(store.stems))
        if missing:
            raise KeyError(f"Requested stems are not aligned in the dataset: {missing}")
        return list(dict.fromkeys(args.stems))
    if args.all_tiles:
        return list(store.stems)
    count = min(args.num_tiles, len(store))
    return random.Random(args.sample_seed).sample(list(store.stems), count)


def _select_candidates(
    args: argparse.Namespace, store: ConditionStore
) -> list[CandidateRecord]:
    if args.candidate_manifest:
        candidates = load_candidate_manifest(
            args.candidate_manifest, available_stems=store.stems
        )
        if args.shard_index is not None:
            candidates = select_candidate_shard(
                candidates,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
        return candidates

    stems = _select_stems(args, store)
    seeds = args.seeds or [42]
    return [
        CandidateRecord(candidate_id=f"{stem}__seed_{seed:010d}", stem=stem, seed=seed)
        for stem in stems
        for seed in seeds
    ]


def _difference_summary(
    original: ConditionBundle, converted: ConditionBundle
) -> dict[str, Any]:
    spatial_delta = converted.spatial - original.spatial
    morphology_delta = converted.morphology - original.morphology
    changed_channels = (
        torch.nonzero(spatial_delta.abs().flatten(1).amax(dim=1) > 0, as_tuple=False)
        .flatten()
        .tolist()
    )
    changed_features = (
        torch.nonzero(morphology_delta.abs() > 0, as_tuple=False).flatten().tolist()
    )
    return {
        "spatial_l2": float(torch.linalg.vector_norm(spatial_delta)),
        "morphology_l2": float(torch.linalg.vector_norm(morphology_delta)),
        "changed_spatial_channels": changed_channels,
        "changed_morphology_features": changed_features,
    }


def _apply_interventions(
    *,
    original: ConditionBundle,
    candidate: CandidateRecord,
    interventions: list[Any],
    store: ConditionStore,
    intervention_seed: int,
) -> list[tuple[Any, Any, dict[str, Any]]]:
    applied_items = []
    for intervention in interventions:
        context = InterventionContext(
            store=store,
            original_stem=candidate.stem,
            intervention_seed=intervention_seed,
            generation_seed=candidate.seed,
        )
        applied = intervention.apply(original, context)
        applied_items.append(
            (
                intervention,
                applied,
                _difference_summary(original, applied.condition),
            )
        )
    return applied_items


def _default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPOSITORY_ROOT / "data/evaluations" / f"counterfactual_{timestamp}"


def main() -> None:
    args = parse_args()
    module, interventions = load_interventions(args.experiment)
    interventions = select_interventions(interventions, args.interventions)
    if args.list_interventions:
        for intervention in interventions:
            print(f"{intervention.slug}\t{intervention.name}")
        return

    resolved_data_root = args.data_root.expanduser().resolve()
    images_dir = args.images_dir
    if images_dir is None and resolved_data_root == DEFAULT_DATA_ROOT.resolve():
        images_dir = DEFAULT_IMAGES_DIR
    store = ConditionStore(
        args.data_root,
        spatial_maps_dir=args.spatial_maps_dir,
        morphology_table=args.morphology_table,
        images_dir=images_dir,
    )
    candidates = _select_candidates(args, store)
    output_dir = (args.output_dir or _default_output_dir()).expanduser().resolve()
    has_existing_output = output_dir.exists() and any(output_dir.iterdir())
    pairs_path = output_dir / "pairs.jsonl"
    images_manifest_path = output_dir / "images.csv"

    experiment_path = Path(module.__file__).resolve()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "dry_run" if args.dry_run else "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": _git_metadata(),
        "versions": _versions(),
        "experiment": {
            "module": module.__name__,
            "source": str(experiment_path),
            "source_sha256": _file_sha256(experiment_path),
            "interventions": [item.manifest() for item in interventions],
        },
        "data": {
            "root": str(store.data_root),
            "spatial_maps_dir": str(store.spatial_maps_dir),
            "morphology_table": str(store.morphology_table),
            "images_dir": str(store.images_dir),
            "aligned_tile_count": len(store),
            "candidate_manifest": (
                str(args.candidate_manifest.expanduser().resolve())
                if args.candidate_manifest
                else None
            ),
            "candidate_manifest_sha256": (
                _file_sha256(args.candidate_manifest.expanduser().resolve())
                if args.candidate_manifest
                else None
            ),
            "candidate_count": len(candidates),
            "candidates": [
                {
                    "candidate_id": item.candidate_id,
                    "stem": item.stem,
                    "seed": item.seed,
                }
                for item in candidates
            ],
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
        },
        "generation": {
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "base_model": args.base_model,
            "revision": args.revision,
            "prompt": args.prompt,
            "seed_source": (
                "candidate_manifest" if args.candidate_manifest else "command_line"
            ),
            "intervention_seed": args.intervention_seed,
            "num_inference_steps": args.steps,
            "spatial_strength": args.spatial_strength,
            "batch_size": args.batch_size,
            "batch_size_history": [args.batch_size],
            "requested_device": args.device,
            "requested_dtype": args.dtype,
            "matched_initial_noise": True,
            "baseline_generated": not args.omit_baseline,
        },
    }
    signature_payload = {
        "experiment_source_sha256": manifest["experiment"]["source_sha256"],
        "interventions": manifest["experiment"]["interventions"],
        "candidate_manifest_sha256": manifest["data"]["candidate_manifest_sha256"],
        "candidates": manifest["data"]["candidates"],
        "data_root": manifest["data"]["root"],
        "checkpoint": manifest["generation"]["checkpoint"],
        "base_model": args.base_model,
        "revision": args.revision,
        "prompt": args.prompt,
        "intervention_seed": args.intervention_seed,
        "num_inference_steps": args.steps,
        "spatial_strength": args.spatial_strength,
        "baseline_generated": not args.omit_baseline,
    }
    manifest["run_signature"] = _payload_sha256(signature_payload)
    existing_status: str | None = None
    if has_existing_output:
        if not args.resume:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}; use --resume only "
                "for the same interrupted run"
            )
        manifest_path = output_dir / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume without an existing run manifest: {manifest_path}"
            )
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("run_signature") != manifest["run_signature"]:
            raise ValueError(
                "Refusing to resume: experiment, cohort, controls, or generation "
                "settings differ from the existing run"
            )
        manifest = existing_manifest
        existing_status = manifest.get("status")
        manifest.setdefault("resumed_at", []).append(
            datetime.now(timezone.utc).isoformat()
        )
        history = manifest["generation"].setdefault("batch_size_history", [])
        if not history or history[-1] != args.batch_size:
            history.append(args.batch_size)
        manifest["generation"]["batch_size"] = args.batch_size
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    _json_write(output_dir / "run_manifest.json", manifest)

    if args.dry_run:
        dry_run_results = {}
        for candidate in candidates:
            original = store.load(candidate.stem)
            applied_items = _apply_interventions(
                original=original,
                candidate=candidate,
                interventions=interventions,
                store=store,
                intervention_seed=args.intervention_seed,
            )
            dry_run_results[candidate.candidate_id] = {
                "stem": candidate.stem,
                "seed": candidate.seed,
                "interventions": [
                    {
                        "intervention": intervention.slug,
                        "details": applied.details,
                        "difference": difference,
                    }
                    for intervention, applied, difference in applied_items
                ],
            }
        manifest["dry_run_results"] = dry_run_results
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        _json_write(output_dir / "run_manifest.json", manifest)
        print(f"Dry run validated {len(interventions)} intervention(s): {output_dir}")
        return

    if args.resume and existing_status == "completed":
        expected_images = len(candidates) * (
            len(interventions) + int(not args.omit_baseline)
        )
        expected_pairs = len(candidates) * len(interventions)
        completed_images = _completed_image_keys(images_manifest_path)
        completed_pairs = _completed_pair_keys(pairs_path)
        if (
            len(completed_images) == expected_images
            and len(completed_pairs) == expected_pairs
        ):
            print(f"Run is already complete; nothing to resume: {output_dir}")
            return

    manifest["status"] = "running"
    _json_write(output_dir / "run_manifest.json", manifest)

    from cpathogen.generation.checkpoints import load_phase2_generation_models
    from cpathogen.generation.counterfactuals import generate_matched_conditions

    models = load_phase2_generation_models(
        args.checkpoint,
        base_model=args.base_model,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    manifest["generation"]["resolved_device"] = str(models.device)
    manifest["generation"]["resolved_dtype"] = str(models.dtype)
    manifest["generation"]["resolved_base_model"] = models.base_model
    _json_write(output_dir / "run_manifest.json", manifest)

    completed_images = _completed_image_keys(images_manifest_path)
    completed_pairs = _completed_pair_keys(pairs_path)
    image_count = len(completed_images)
    pair_count = len(completed_pairs)
    expected_image_count = len(candidates) * (
        len(interventions) + int(not args.omit_baseline)
    )
    expected_pair_count = len(candidates) * len(interventions)
    pending_jobs: list[_GenerationJob] = []
    generated_batches = 0

    manifest["generation"]["batching_strategy"] = (
        "cross-candidate condition batches with per-condition seed replay"
    )
    manifest["generation"]["matched_initial_noise"] = True
    _json_write(output_dir / "run_manifest.json", manifest)

    def append_pair_record(job: _GenerationJob) -> None:
        nonlocal pair_count
        if job.intervention is None or job.applied is None:
            raise RuntimeError("Pair records require intervention metadata")
        record = MatchedPairRecord(
            candidate_id=job.candidate.candidate_id,
            stem=job.candidate.stem,
            seed=job.candidate.seed,
            prompt=args.prompt,
            baseline_image=(
                str(job.baseline_path) if job.baseline_path is not None else None
            ),
            counterfactual_image=str(job.output_path),
            reference_tile=(
                str(job.reference_tile) if job.reference_tile is not None else None
            ),
            intervention=job.intervention.manifest(),
            applied_details=job.applied.details,
            difference=job.difference or {},
        )
        _append_jsonl(pairs_path, record.to_dict())
        completed_pairs.add(
            (job.candidate.candidate_id, job.intervention.slug)
        )
        pair_count = len(completed_pairs)

    def flush_jobs() -> None:
        nonlocal image_count, generated_batches
        if not pending_jobs:
            return
        jobs = list(pending_jobs)
        pending_jobs.clear()
        images = generate_matched_conditions(
            models,
            [job.condition for job in jobs],
            seed=0,
            prompt=args.prompt,
            num_inference_steps=args.steps,
            spatial_strength=args.spatial_strength,
            matched_noise=False,
            per_condition_seeds=[job.candidate.seed for job in jobs],
        )
        for job, image in zip(jobs, images, strict=True):
            _save_png(image, job.output_path)
            parameters = (
                "{}"
                if job.intervention is None
                else json.dumps(job.intervention.parameters(), sort_keys=True)
            )
            _append_image_manifest(
                images_manifest_path,
                {
                    "candidate_id": job.candidate.candidate_id,
                    "stem": job.candidate.stem,
                    "seed": job.candidate.seed,
                    "condition": job.condition_name,
                    "intervention_parameters": parameters,
                    "image_path": str(job.output_path),
                },
            )
            completed_images.add(
                (job.candidate.candidate_id, job.condition_name)
            )
            if job.intervention is not None and not job.pair_already_recorded:
                append_pair_record(job)
        image_count = len(completed_images)
        generated_batches += 1
        manifest["progress"] = {
            "image_count": image_count,
            "expected_image_count": expected_image_count,
            "pair_count": pair_count,
            "expected_pair_count": expected_pair_count,
            "generated_batches_this_process": generated_batches,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _json_write(output_dir / "run_manifest.json", manifest)
        print(
            f"[generation] {image_count}/{expected_image_count} images; "
            f"{pair_count}/{expected_pair_count} pairs",
            flush=True,
        )

    def queue_job(job: _GenerationJob) -> None:
        pending_jobs.append(job)
        if len(pending_jobs) >= args.batch_size:
            flush_jobs()

    for candidate in candidates:
        stem = candidate.stem
        seed = candidate.seed
        original = store.load(stem)
        applied_items = _apply_interventions(
            original=original,
            candidate=candidate,
            interventions=interventions,
            store=store,
            intervention_seed=args.intervention_seed,
        )
        seed_dir = output_dir / "images" / candidate.candidate_id / f"seed_{seed:010d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        baseline_file = seed_dir / "baseline.png"
        baseline_path = None if args.omit_baseline else baseline_file
        reference_tile = store.image_path(stem)

        baseline_key = (candidate.candidate_id, "baseline")
        if not args.omit_baseline and baseline_key not in completed_images:
            queue_job(
                _GenerationJob(
                    candidate=candidate,
                    condition_name="baseline",
                    condition=original,
                    output_path=baseline_file,
                    baseline_path=baseline_file,
                    reference_tile=reference_tile,
                )
            )

        for intervention, applied, difference in applied_items:
            key = (candidate.candidate_id, intervention.slug)
            counterfactual_path = seed_dir / f"{intervention.slug}.png"
            pair_exists = key in completed_pairs
            job = _GenerationJob(
                candidate=candidate,
                condition_name=intervention.slug,
                condition=applied.condition,
                output_path=counterfactual_path,
                baseline_path=baseline_path,
                intervention=intervention,
                applied=applied,
                difference=difference,
                reference_tile=reference_tile,
                pair_already_recorded=pair_exists,
            )
            if key in completed_images:
                if not pair_exists:
                    append_pair_record(job)
                continue
            queue_job(job)

    flush_jobs()

    if image_count != expected_image_count or pair_count != expected_pair_count:
        raise RuntimeError(
            "Generation ended with incomplete manifests: "
            f"images={image_count}/{expected_image_count}, "
            f"pairs={pair_count}/{expected_pair_count}"
        )
    manifest["status"] = "completed"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["pair_count"] = pair_count
    manifest["image_count"] = image_count
    manifest["pairs_manifest"] = str(pairs_path)
    manifest["images_manifest"] = str(images_manifest_path)
    _json_write(output_dir / "run_manifest.json", manifest)
    print(f"Generated {pair_count} matched counterfactual pair(s): {output_dir}")


if __name__ == "__main__":
    main()
