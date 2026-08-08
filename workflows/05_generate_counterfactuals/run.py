#!/usr/bin/env python3
"""Generate matched-noise baseline/counterfactual images from Python plugins."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from cpathogen.counterfactuals import (
    ConditionBundle,
    ConditionStore,
    InterventionContext,
    MatchedPairRecord,
    load_interventions,
    select_interventions,
)

DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data/processed/conditions"
DEFAULT_IMAGES_DIR = REPOSITORY_ROOT / "data/interim/tiles/tcga_brca"
DEFAULT_CHECKPOINT = (
    REPOSITORY_ROOT / "artifacts/models/pathogen_phase2/checkpoint_30000"
)


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
    parser.add_argument("--num-tiles", type=int, default=1)
    parser.add_argument(
        "--all-tiles",
        action="store_true",
        help="Generate for every stem with both an aligned spatial map and morphology row.",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--intervention-seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
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
    args = parser.parse_args()
    if args.num_tiles < 1:
        parser.error("--num-tiles must be at least 1")
    if args.all_tiles and args.stems:
        parser.error("--all-tiles cannot be combined with --stem")
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _default_output_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPOSITORY_ROOT / "artifacts/runs/counterfactual_generation" / timestamp


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
    stems = _select_stems(args, store)
    seeds = args.seeds or [42]
    output_dir = (args.output_dir or _default_output_dir()).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = output_dir / "pairs.jsonl"

    experiment_path = Path(module.__file__).resolve()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "dry_run" if args.dry_run else "running",
        "created_at": datetime.now(UTC).isoformat(),
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
            "selected_stems": stems,
        },
        "generation": {
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "base_model": args.base_model,
            "revision": args.revision,
            "prompt": args.prompt,
            "seeds": seeds,
            "intervention_seed": args.intervention_seed,
            "num_inference_steps": args.steps,
            "batch_size": args.batch_size,
            "requested_device": args.device,
            "requested_dtype": args.dtype,
            "matched_initial_noise": True,
        },
    }
    _json_write(output_dir / "run_manifest.json", manifest)

    applied_by_stem: dict[str, list[tuple[Any, Any, dict[str, Any]]]] = {}
    for stem in stems:
        original = store.load(stem)
        applied_items = []
        for intervention in interventions:
            context = InterventionContext(
                store=store,
                original_stem=stem,
                intervention_seed=args.intervention_seed,
                generation_seed=seeds[0],
            )
            applied = intervention.apply(original, context)
            applied_items.append(
                (
                    intervention,
                    applied,
                    _difference_summary(original, applied.condition),
                )
            )
        applied_by_stem[stem] = applied_items

    if args.dry_run:
        manifest["dry_run_results"] = {
            stem: [
                {
                    "intervention": intervention.slug,
                    "details": applied.details,
                    "difference": difference,
                }
                for intervention, applied, difference in applied_by_stem[stem]
            ]
            for stem in stems
        }
        manifest["completed_at"] = datetime.now(UTC).isoformat()
        _json_write(output_dir / "run_manifest.json", manifest)
        print(f"Dry run validated {len(interventions)} intervention(s): {output_dir}")
        return

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

    pair_count = 0
    for stem in stems:
        original = store.load(stem)
        applied_items = applied_by_stem[stem]
        for seed in seeds:
            seed_dir = output_dir / "images" / stem / f"seed_{seed:010d}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            baseline_path = seed_dir / "baseline.png"
            pending = list(applied_items)
            first_capacity = max(0, args.batch_size - 1)
            first_items = pending[:first_capacity]
            pending = pending[first_capacity:]
            first_conditions = [original] + [item[1].condition for item in first_items]
            first_images = generate_matched_conditions(
                models,
                first_conditions,
                seed=seed,
                prompt=args.prompt,
                num_inference_steps=args.steps,
                matched_noise=True,
            )
            first_images[0].save(baseline_path)
            generated_groups = [(first_items, first_images[1:])]
            while pending:
                group = pending[: args.batch_size]
                pending = pending[args.batch_size :]
                images = generate_matched_conditions(
                    models,
                    [item[1].condition for item in group],
                    seed=seed,
                    prompt=args.prompt,
                    num_inference_steps=args.steps,
                    matched_noise=True,
                )
                generated_groups.append((group, images))

            for group, images in generated_groups:
                for (intervention, applied, difference), image in zip(
                    group, images, strict=True
                ):
                    counterfactual_path = seed_dir / f"{intervention.slug}.png"
                    image.save(counterfactual_path)
                    reference_tile = store.image_path(stem)
                    record = MatchedPairRecord(
                        stem=stem,
                        seed=seed,
                        prompt=args.prompt,
                        baseline_image=str(baseline_path),
                        counterfactual_image=str(counterfactual_path),
                        reference_tile=str(reference_tile) if reference_tile else None,
                        intervention=intervention.manifest(),
                        applied_details=applied.details,
                        difference=difference,
                    )
                    _append_jsonl(pairs_path, record.to_dict())
                    pair_count += 1
            if models.device.type == "mps":
                torch.mps.empty_cache()
            elif models.device.type == "cuda":
                torch.cuda.empty_cache()

    manifest["status"] = "completed"
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    manifest["pair_count"] = pair_count
    manifest["pairs_manifest"] = str(pairs_path)
    _json_write(output_dir / "run_manifest.json", manifest)
    print(f"Generated {pair_count} matched counterfactual pair(s): {output_dir}")


if __name__ == "__main__":
    main()
