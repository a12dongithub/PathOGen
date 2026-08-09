#!/usr/bin/env python
"""Compare baseline FID/KID with CellViT++ spatial reranking at fixed 30 steps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.colab.layout import DEFAULT_CONFIG, RuntimePaths
from experiments.fidelity.constants import MORPH_FEATURES
from experiments.fidelity.data import CellObservation, DatasetCatalog, load_cells
from experiments.fidelity.guidance import GenerationContext
from experiments.fidelity.workflow import (
    ExperimentRuntime,
    deterministic_seed,
    write_json,
)


@dataclass(frozen=True)
class CandidateConfig:
    config_id: str
    green_sd: float
    controlnet_strength: float
    denoising_steps: int


@dataclass
class CandidateJob:
    input_index: int
    stem: str
    candidate_order: int
    config: CandidateConfig
    seed_index: int
    context: GenerationContext
    green_details: dict[str, Any]
    artifact_name: str
    generated_result: tuple[Path, dict[str, Any]] | None = None
    cellvit_geojson: Path | None = None


# Focused design after the first 100-case pilot: spatial-conditioning strength is
# fixed at 2, denoising at 30 steps, and only neutral/-1 SD green are searched.
DEFAULT_CONFIGS = (
    CandidateConfig("cfg00_g0_c2_s30", 0.0, 2.0, 30),
    CandidateConfig("cfg01_gm1_c2_s30", -1.0, 2.0, 30),
)
DEFAULT_SEEDS_PER_CONFIG = 8
BASELINE_CONFIG_ID = "cfg00_g0_c2_s30"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--cellvit-root", type=Path)
    parser.add_argument("--cellvit-model", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--stems", nargs="*")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds-per-config",
        type=int,
        default=DEFAULT_SEEDS_PER_CONFIG,
        help="Deterministic noise seeds evaluated for each parameter configuration",
    )
    parser.add_argument("--match-radius", type=float, default=50.0)
    parser.add_argument("--green-lower-quantile", type=float, default=0.01)
    parser.add_argument("--green-upper-quantile", type=float, default=0.99)
    parser.add_argument(
        "--generator-precision", choices=("auto", "fp16", "fp32"), default="auto"
    )
    parser.add_argument(
        "--generator-memory-mode",
        choices=("auto", "throughput", "balanced", "low-vram"),
        default="auto",
        help="GPU memory strategy; auto uses throughput mode on GPUs with >=20 GiB",
    )
    parser.add_argument(
        "--cellvit-precision", choices=("auto", "fp16", "fp32"), default="auto"
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=4,
        help=(
            "Maximum PathOGen batch across source cases; automatically halved if "
            "CUDA runs out of memory"
        ),
    )
    parser.add_argument(
        "--cellvit-batch-size",
        type=int,
        default=4,
        help=(
            "Maximum CellViT++ batch across source cases; automatically halved if "
            "CUDA runs out of memory"
        ),
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Rewrite resumable score CSVs after this many completed inputs",
    )
    parser.add_argument("--kid-subset-size", type=int, default=100)
    parser.add_argument("--kid-subsets", type=int, default=100)
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Calculate baseline FID/KID and stop before candidate reranking",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Require and reuse the cached baseline metric/images, then start reranking",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Development-only option; normal experiment runs always calculate FID and KID",
    )
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    if config_path.is_file():
        paths = RuntimePaths.read(config_path)
        args.data_dir = args.data_dir or paths.data_dir
        args.checkpoint_dir = args.checkpoint_dir or paths.checkpoint_dir
        args.cellvit_root = args.cellvit_root or paths.cellvit_root
        args.cellvit_model = args.cellvit_model or paths.cellvit_model
        args.output_dir = (
            args.output_dir or paths.output_root / "cellvit_rerank_fid_kid"
        )
    missing = [
        name
        for name in (
            "data_dir",
            "checkpoint_dir",
            "cellvit_root",
            "cellvit_model",
            "output_dir",
        )
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(
            f"Missing paths {missing}. Run experiments/colab/setup_colab.py or pass them explicitly."
        )
    # ExperimentRuntime expects these common workflow attributes. Candidate-specific
    # steps and strength are passed explicitly for every generation.
    args.steps = 20
    args.spatial_strength = 1.0
    args.guidance_hook = None
    args.guidance_config = None
    args.max_guidance_attempts = 1
    args.keep_rejected = False
    return args


def _bounded_assignment(
    source_indices: list[int],
    predicted_indices: list[int],
    source: list[CellObservation],
    predicted: list[CellObservation],
    radius: float,
    same_type: bool,
) -> list[tuple[int, int, float]]:
    if not source_indices or not predicted_indices:
        return []
    source_xy = np.asarray(
        [source[index].centroid for index in source_indices], dtype=float
    )
    predicted_xy = np.asarray(
        [predicted[index].centroid for index in predicted_indices], dtype=float
    )
    distances = np.linalg.norm(source_xy[:, None, :] - predicted_xy[None, :, :], axis=2)
    type_agreement = np.asarray(
        [
            [source[i].cell_type == predicted[j].cell_type for j in predicted_indices]
            for i in source_indices
        ],
        dtype=bool,
    )
    valid = distances <= radius
    valid &= type_agreement if same_type else ~type_agreement

    # Each target receives either one unique valid prediction or a dummy unmatched
    # column. A valid edge always costs less than a dummy, so assignment first
    # maximizes match count and then minimizes distance.
    source_count, predicted_count = distances.shape
    cost = np.full((source_count, predicted_count + source_count), 2.0)
    cost[:, :predicted_count] = np.where(
        valid, distances / max(radius, 1e-6), 1_000_000.0
    )
    rows, columns = linear_sum_assignment(cost)
    matches = []
    for row, column in zip(rows.tolist(), columns.tolist()):
        if column < predicted_count and valid[row, column]:
            matches.append(
                (
                    source_indices[row],
                    predicted_indices[column],
                    float(distances[row, column]),
                )
            )
    return matches


def score_spatial_cells(
    source: list[CellObservation],
    predicted: list[CellObservation],
    radius: float = 50.0,
) -> dict[str, float | int]:
    """Apply the requested +1/0/-1 point score with unique cell matching.

    Same-type prediction within radius: +1. Different-type prediction within
    radius: 0. No prediction within radius: -1. Each prediction can match at most
    one input cell. Extra predictions are reported but do not alter the score.
    """

    if radius <= 0:
        raise ValueError("match radius must be positive")
    remaining_source = list(range(len(source)))
    remaining_predicted = list(range(len(predicted)))
    same_matches = _bounded_assignment(
        remaining_source,
        remaining_predicted,
        source,
        predicted,
        radius,
        same_type=True,
    )
    used_source = {match[0] for match in same_matches}
    used_predicted = {match[1] for match in same_matches}
    remaining_source = [index for index in remaining_source if index not in used_source]
    remaining_predicted = [
        index for index in remaining_predicted if index not in used_predicted
    ]
    wrong_matches = _bounded_assignment(
        remaining_source,
        remaining_predicted,
        source,
        predicted,
        radius,
        same_type=False,
    )
    used_source.update(match[0] for match in wrong_matches)
    used_predicted.update(match[1] for match in wrong_matches)

    same_count = len(same_matches)
    wrong_count = len(wrong_matches)
    missing_count = len(source) - same_count - wrong_count
    raw_score = same_count - missing_count
    normalized_score = raw_score / len(source) if source else 0.0
    same_distances = [match[2] for match in same_matches]
    return {
        "spatial_score": float(normalized_score),
        "spatial_points": int(raw_score),
        "target_cells": len(source),
        "predicted_cells": len(predicted),
        "same_type_matches": same_count,
        "different_type_matches": wrong_count,
        "missing_matches": missing_count,
        "extra_predictions": len(predicted) - len(used_predicted),
        "mean_same_type_distance": (
            float(np.mean(same_distances)) if same_distances else float("nan")
        ),
    }


def green_condition(
    morphology: np.ndarray,
    green_sd: float,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    index = MORPH_FEATURES.index("g_mean")
    changed = np.asarray(morphology, dtype=np.float32).copy()
    requested = float(changed[index] + green_sd)
    changed[index] = np.clip(requested, lower, upper)
    return changed, {
        "green_baseline": float(morphology[index]),
        "green_requested": requested,
        "green_applied": float(changed[index]),
        "green_clipped": not math.isclose(
            requested, float(changed[index]), abs_tol=1e-7
        ),
    }


def run_id(
    stems: list[str],
    seed: int,
    radius: float,
    green_lower_quantile: float,
    green_upper_quantile: float,
    checkpoint_dir: Path,
    seeds_per_config: int,
) -> str:
    payload = {
        "stems": stems,
        "seed": seed,
        "radius": radius,
        "green_quantiles": [green_lower_quantile, green_upper_quantile],
        "checkpoint_dir": str(checkpoint_dir.expanduser().resolve()),
        "configs": [asdict(config) for config in DEFAULT_CONFIGS],
        "seeds_per_config": seeds_per_config,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[
        :12
    ]
    return f"cellvit_rerank_{digest}"


def copy_image(source: Path, destination: Path, overwrite: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not destination.is_file():
        with Image.open(source) as image:
            image.convert("RGB").save(destination)


def calculate_fid_kid(
    real_dir: Path,
    generated_dir: Path,
    subset_size: int,
    subsets: int,
    rng_seed: int,
) -> dict[str, float | int]:
    try:
        import torch
        import torch_fidelity
    except ImportError as error:
        raise RuntimeError(
            "FID/KID requires torch-fidelity. Run `pip install -r "
            "experiments/requirements_fidelity.txt`."
        ) from error

    real_count = len(list(real_dir.glob("*.png")))
    generated_count = len(list(generated_dir.glob("*.png")))
    if real_count != generated_count:
        raise RuntimeError(
            f"FID/KID set sizes differ: real={real_count}, generated={generated_count}"
        )
    if real_count < 2:
        raise ValueError("FID/KID requires at least two real and generated images")
    actual_subset_size = min(int(subset_size), real_count)
    metrics = torch_fidelity.calculate_metrics(
        input1=str(real_dir),
        input2=str(generated_dir),
        cuda=torch.cuda.is_available(),
        isc=False,
        fid=True,
        kid=True,
        kid_subset_size=actual_subset_size,
        kid_subsets=int(subsets),
        rng_seed=int(rng_seed),
        verbose=False,
    )
    return {
        "images": real_count,
        "fid": max(0.0, float(metrics["frechet_inception_distance"])),
        "kid_mean": float(metrics["kernel_inception_distance_mean"]),
        "kid_std": float(metrics["kernel_inception_distance_std"]),
        "kid_subset_size": actual_subset_size,
        "kid_subsets": int(subsets),
        "rng_seed": int(rng_seed),
    }


def context_for_candidate(
    runtime: ExperimentRuntime,
    stem: str,
    config: CandidateConfig,
    seed_index: int,
    green_lower: float,
    green_upper: float,
) -> tuple[GenerationContext, dict[str, Any]]:
    sample = runtime.catalog.sample(stem)
    morphology, green_details = green_condition(
        sample.morphology, config.green_sd, green_lower, green_upper
    )
    candidate_seed = deterministic_seed(
        runtime.args.seed, stem, "cellvit_rerank", f"seed_{seed_index:02d}"
    )
    context = GenerationContext(
        stem=stem,
        condition_id=f"{config.config_id}__seed{seed_index:02d}",
        spatial_map=runtime.catalog.load_spatial(sample.spatial_path),
        morphology=morphology,
        seed=candidate_seed,
        metadata={
            "experiment": "cellvit_spatial_rerank_fid_kid",
            "candidate_config": asdict(config),
            "seed_index": seed_index,
            **green_details,
        },
    )
    return context, green_details


def artifact_name(stem: str, config: CandidateConfig, seed_index: int) -> str:
    return f"{stem}__rerank__{config.config_id}__seed{seed_index:02d}"


def batches(values: list[Any], batch_size: int) -> list[list[Any]]:
    return [
        values[start : start + batch_size]
        for start in range(0, len(values), batch_size)
    ]


def rerank_input_window_size(
    configs: tuple[CandidateConfig, ...],
    seeds_per_config: int,
    generation_batch_size: int,
    cellvit_batch_size: int,
) -> int:
    """Choose enough source cases to fill both candidate-model batches.

    Generation calls require common denoising steps and spatial strength, while
    CellViT++ can consume every candidate together. The returned window is only a
    scheduling unit; each model call is still capped by its requested batch size.
    """

    generation_groups = Counter(
        (config.denoising_steps, config.controlnet_strength) for config in configs
    )
    candidates_per_input = len(configs) * seeds_per_config
    generation_inputs = max(
        math.ceil(generation_batch_size / (config_count * seeds_per_config))
        for config_count in generation_groups.values()
    )
    cellvit_inputs = math.ceil(cellvit_batch_size / candidates_per_input)
    return max(1, generation_inputs, cellvit_inputs)


def build_candidate_jobs(
    runtime: ExperimentRuntime,
    stems: list[str],
    first_input_index: int,
    seeds_per_config: int,
    green_lower: float,
    green_upper: float,
) -> list[CandidateJob]:
    jobs = []
    for stem_offset, stem in enumerate(stems):
        input_index = first_input_index + stem_offset
        for config_index, config in enumerate(DEFAULT_CONFIGS):
            for seed_index in range(seeds_per_config):
                context, green_details = context_for_candidate(
                    runtime,
                    stem,
                    config,
                    seed_index,
                    green_lower,
                    green_upper,
                )
                jobs.append(
                    CandidateJob(
                        input_index=input_index,
                        stem=stem,
                        candidate_order=config_index * seeds_per_config + seed_index,
                        config=config,
                        seed_index=seed_index,
                        context=context,
                        green_details=green_details,
                        artifact_name=artifact_name(stem, config, seed_index),
                    )
                )
    return jobs


def load_rerank_progress(
    experiment_dir: Path,
    stems: list[str],
    seeds_per_config: int,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Resume from the last fully checkpointed input prefix."""
    if overwrite:
        return [], [], 0
    scores_path = experiment_dir / "candidate_scores.csv"
    selections_path = experiment_dir / "selected_candidates.csv"
    if not scores_path.is_file() or not selections_path.is_file():
        return [], [], 0
    try:
        score_frame = pd.read_csv(scores_path)
        selection_frame = pd.read_csv(selections_path)
    except (OSError, pd.errors.ParserError) as error:
        print(f"[resume] Could not read progress CSVs; rebuilding: {error}", flush=True)
        return [], [], 0
    if "stem" not in score_frame or "stem" not in selection_frame:
        print("[resume] Progress CSVs lack stem columns; rebuilding", flush=True)
        return [], [], 0
    completed_stems = selection_frame["stem"].astype(str).tolist()
    expected_rows = len(completed_stems) * len(DEFAULT_CONFIGS) * seeds_per_config
    if (
        completed_stems != stems[: len(completed_stems)]
        or len(score_frame) != expected_rows
    ):
        print(
            "[resume] Progress CSVs do not match the requested stem/config prefix; "
            "rebuilding from artifacts",
            flush=True,
        )
        return [], [], 0
    print(
        f"[resume] Loaded {len(completed_stems)} completed inputs and "
        f"{len(score_frame)} candidate scores",
        flush=True,
    )
    return (
        score_frame.to_dict(orient="records"),
        selection_frame.to_dict(orient="records"),
        len(completed_stems),
    )


def main() -> None:
    args = parse_args()
    if args.num_images < 1:
        raise ValueError("num-images must be positive")
    if not 0 <= args.green_lower_quantile < args.green_upper_quantile <= 1:
        raise ValueError("green quantiles must satisfy 0 <= lower < upper <= 1")
    if args.kid_subsets < 1 or args.kid_subset_size < 2:
        raise ValueError(
            "KID subsets must be positive and subset size must be at least two"
        )
    if args.generation_batch_size < 1 or args.cellvit_batch_size < 1:
        raise ValueError("generation and CellViT batch sizes must be positive")
    if args.seeds_per_config < 1:
        raise ValueError("seeds-per-config must be positive")
    if args.save_every < 1:
        raise ValueError("save-every must be positive")
    if args.baseline_only and args.skip_baseline:
        raise ValueError("baseline-only and skip-baseline cannot be used together")

    catalog = DatasetCatalog(args.data_dir)
    stems = catalog.select(args.num_images, args.seed, args.stems)
    if not args.dry_run and not args.skip_metrics and len(stems) < 2:
        raise ValueError("Use at least two inputs when calculating FID/KID")
    if not args.dry_run and not args.skip_metrics and len(stems) < 1000:
        print(
            f"WARNING: FID from {len(stems)} images is preliminary and statistically unstable; "
            "use a much larger held-out set for paper results. FID values measured with "
            "different sample counts or implementations are not directly comparable.",
            flush=True,
        )
    output_root = args.output_dir.expanduser().resolve()
    experiment_dir = output_root / run_id(
        stems,
        args.seed,
        args.match_radius,
        args.green_lower_quantile,
        args.green_upper_quantile,
        args.checkpoint_dir,
        args.seeds_per_config,
    )
    args.output_dir = experiment_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)
    green_series = catalog.morphology["g_mean"].dropna()
    green_lower = float(green_series.quantile(args.green_lower_quantile))
    green_upper = float(green_series.quantile(args.green_upper_quantile))
    manifest = {
        "experiment": "cellvit_spatial_rerank_fid_kid",
        "runtime_config": str(args.config.expanduser().resolve()),
        "run_directory": str(experiment_dir),
        "stems": stems,
        "num_inputs": len(stems),
        "configs": [asdict(config) for config in DEFAULT_CONFIGS],
        "seeds_per_config": args.seeds_per_config,
        "candidates_per_input": len(DEFAULT_CONFIGS) * args.seeds_per_config,
        "total_candidate_images": len(stems)
        * len(DEFAULT_CONFIGS)
        * args.seeds_per_config,
        "generation_batch_size": args.generation_batch_size,
        "cellvit_batch_size": args.cellvit_batch_size,
        "rerank_batching": "candidates flattened across source inputs",
        "baseline": {"config_id": BASELINE_CONFIG_ID, "seed_index": 0},
        "baseline_only": args.baseline_only,
        "skip_baseline": args.skip_baseline,
        "green_allowed_standardized_range": [green_lower, green_upper],
        "score": {
            "input_cells": "exact source GeoJSON centroids used to create the conditioning map",
            "generated_cells": "CellViT++ centroids and types from each generated image",
            "radius_pixels": args.match_radius,
            "same_type_within_radius": 1,
            "different_type_within_radius": 0,
            "no_cell_within_radius": -1,
            "unique_prediction_per_target": True,
            "extra_prediction_penalty": 0,
            "tie_break": "first candidate in fixed config/seed order",
        },
        "fid_kid_warning": (
            "FID is upward-biased and unstable for small num-images. Compare models only "
            "with identical sample count, real subset, preprocessing, and implementation."
        ),
    }
    write_json(experiment_dir / "experiment_manifest.json", manifest)
    if args.dry_run:
        for stem in stems:
            sample = catalog.sample(stem)
            baseline = sample.morphology
            if not load_cells(sample.geojson_path):
                raise RuntimeError(f"Source GeoJSON has no recognized cells: {stem}")
            for config in DEFAULT_CONFIGS:
                changed, _ = green_condition(
                    baseline, config.green_sd, green_lower, green_upper
                )
                changed_indices = np.flatnonzero(
                    ~np.isclose(changed, baseline, atol=1e-7, rtol=0)
                )
                if config.green_sd == 0 and len(changed_indices):
                    raise AssertionError(
                        "Green-neutral configuration changed morphology"
                    )
                if config.green_sd != 0 and changed_indices.tolist() not in (
                    [],
                    [MORPH_FEATURES.index("g_mean")],
                ):
                    raise AssertionError(
                        "Green intervention changed a non-green feature"
                    )
        print(
            f"Dry run passed: {len(stems)} inputs x {len(DEFAULT_CONFIGS)} configs "
            f"x {args.seeds_per_config} seeds = "
            f"{len(stems) * len(DEFAULT_CONFIGS) * args.seeds_per_config} candidates"
        )
        return

    real_dir = experiment_dir / "metric_sets" / "real"
    baseline_dir = experiment_dir / "metric_sets" / "baseline"
    selected_dir = experiment_dir / "metric_sets" / "selected"
    baseline_config = next(
        config for config in DEFAULT_CONFIGS if config.config_id == BASELINE_CONFIG_ID
    )

    # Phase 1: generate exactly one fixed baseline candidate per condition, then
    # calculate baseline FID/KID before any CellViT++ selection. A reset Colab
    # runtime can bypass this entire phase when the Drive-backed cache is complete.
    baseline_metrics_path = experiment_dir / "metrics_before_reranking.json"
    baseline_metrics = None
    if args.skip_baseline:
        real_count = len(list(real_dir.glob("*.png")))
        baseline_count = len(list(baseline_dir.glob("*.png")))
        if real_count != len(stems) or baseline_count != len(stems):
            raise RuntimeError(
                "Cannot skip baseline: cached metric sets are incomplete "
                f"(expected={len(stems)}, real={real_count}, baseline={baseline_count})"
            )
        if not args.skip_metrics:
            if not baseline_metrics_path.is_file():
                raise FileNotFoundError(
                    f"Cannot skip baseline: cached metrics missing: {baseline_metrics_path}"
                )
            baseline_metrics = json.loads(
                baseline_metrics_path.read_text(encoding="utf-8")
            )
            if int(baseline_metrics.get("images", -1)) != len(stems):
                raise RuntimeError(
                    "Cannot skip baseline: cached metric image count does not match "
                    f"the requested {len(stems)} inputs"
                )
        print(f"[baseline skipped] reused {len(stems)} cached image pairs", flush=True)
        if baseline_metrics is not None:
            print(f"[metrics before reused] {baseline_metrics}", flush=True)
    else:
        phase_one = ExperimentRuntime(args)
        baseline_rows = []
        try:
            completed = 0
            for stem_batch in batches(stems, args.generation_batch_size):
                entries = []
                for stem in stem_batch:
                    sample = phase_one.catalog.sample(stem)
                    context, green_details = context_for_candidate(
                        phase_one, stem, baseline_config, 0, green_lower, green_upper
                    )
                    entries.append(
                        (
                            stem,
                            sample,
                            context,
                            green_details,
                            artifact_name(stem, baseline_config, 0),
                        )
                    )
                generated = phase_one.ensure_generated_batch(
                    [entry[2] for entry in entries],
                    [entry[4] for entry in entries],
                    steps=baseline_config.denoising_steps,
                    spatial_strength=baseline_config.controlnet_strength,
                )
                for entry, (image_path, generation_metadata) in zip(entries, generated):
                    stem, sample, context, green_details, _ = entry
                    copy_image(
                        sample.image_path, real_dir / f"{stem}.png", args.overwrite
                    )
                    copy_image(image_path, baseline_dir / f"{stem}.png", args.overwrite)
                    baseline_rows.append(
                        {
                            "stem": stem,
                            "generated_image": str(image_path),
                            "real_image": str(sample.image_path),
                            "seed": context.seed,
                            **asdict(baseline_config),
                            **green_details,
                            "generation_seconds": generation_metadata.get(
                                "seconds", np.nan
                            ),
                        }
                    )
                    completed += 1
                    print(f"[baseline {completed}/{len(stems)}] {stem}", flush=True)
        finally:
            phase_one.close()
        pd.DataFrame(baseline_rows).to_csv(
            experiment_dir / "baseline_images.csv", index=False
        )

    if not args.skip_metrics and baseline_metrics is None:
        if baseline_metrics_path.is_file() and not args.overwrite:
            cached = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
            if int(cached.get("images", -1)) == len(stems):
                baseline_metrics = cached
                print(f"[metrics before reused] {baseline_metrics}", flush=True)
        if baseline_metrics is None:
            baseline_metrics = calculate_fid_kid(
                real_dir,
                baseline_dir,
                args.kid_subset_size,
                args.kid_subsets,
                args.seed,
            )
            write_json(baseline_metrics_path, baseline_metrics)
            print(f"[metrics before] {baseline_metrics}", flush=True)

    if args.baseline_only:
        write_json(
            experiment_dir / "fid_kid_comparison.json",
            {
                "before": baseline_metrics,
                "after": None,
                "improvement_lower_is_better": None,
                "status": "baseline_only",
            },
        )
        print(
            f"Baseline-only run complete; reusable artifacts written to {experiment_dir}",
            flush=True,
        )
        return

    # Phase 2: generate/reuse all candidates, run CellViT++, apply only the
    # requested spatial/type point score, and copy the highest-scoring candidate.
    phase_two = ExperimentRuntime(args)
    all_rows, selections, completed_inputs = load_rerank_progress(
        experiment_dir, stems, args.seeds_per_config, args.overwrite
    )
    input_window_size = rerank_input_window_size(
        DEFAULT_CONFIGS,
        args.seeds_per_config,
        args.generation_batch_size,
        args.cellvit_batch_size,
    )
    print(
        f"[rerank batching] up to {input_window_size} source inputs per window; "
        f"generation batch={args.generation_batch_size}, "
        f"CellViT batch={args.cellvit_batch_size}",
        flush=True,
    )
    try:
        remaining_stems = stems[completed_inputs:]
        processed_inputs = completed_inputs
        for stem_window in batches(remaining_stems, input_window_size):
            first_input_index = processed_inputs + 1
            last_input_index = processed_inputs + len(stem_window)
            jobs = build_candidate_jobs(
                phase_two,
                stem_window,
                first_input_index,
                args.seeds_per_config,
                green_lower,
                green_upper,
            )

            # PathOGen accepts varying morphology and spatial maps in one batch,
            # but steps/ControlNet strength are scalar call arguments. Group only
            # by those settings, then batch candidates from multiple source cases.
            generation_groups: dict[tuple[int, float], list[CandidateJob]] = {}
            for job in jobs:
                key = (job.config.denoising_steps, job.config.controlnet_strength)
                generation_groups.setdefault(key, []).append(job)
            for (steps, strength), generation_jobs in generation_groups.items():
                for job_batch in batches(generation_jobs, args.generation_batch_size):
                    generated = phase_two.ensure_generated_batch(
                        [job.context for job in job_batch],
                        [job.artifact_name for job in job_batch],
                        steps=steps,
                        spatial_strength=strength,
                    )
                    for job, result in zip(job_batch, generated):
                        job.generated_result = result

            for job_batch in batches(jobs, args.cellvit_batch_size):
                if any(job.generated_result is None for job in job_batch):
                    raise RuntimeError(
                        "Internal error: candidate generation is incomplete"
                    )
                geojson_paths = phase_two.ensure_cellvit_batch(
                    [job.generated_result[0] for job in job_batch],  # type: ignore[index]
                    [job.artifact_name for job in job_batch],
                )
                for job, geojson_path in zip(job_batch, geojson_paths):
                    job.cellvit_geojson = geojson_path

            print(
                f"[rerank batch {first_input_index}-{last_input_index}/{len(stems)}] "
                f"generated and segmented {len(jobs)} candidates",
                flush=True,
            )

            for stem in stem_window:
                processed_inputs += 1
                source_cells = load_cells(phase_two.catalog.sample(stem).geojson_path)
                candidate_rows = []
                for job in (candidate for candidate in jobs if candidate.stem == stem):
                    if job.generated_result is None or job.cellvit_geojson is None:
                        raise RuntimeError(
                            "Internal error: candidate inference is incomplete"
                        )
                    image_path, generation_metadata = job.generated_result
                    predicted_cells = load_cells(job.cellvit_geojson)
                    score = score_spatial_cells(
                        source_cells, predicted_cells, args.match_radius
                    )
                    row = {
                        "stem": stem,
                        "candidate_order": job.candidate_order,
                        "seed_index": job.seed_index,
                        "seed": job.context.seed,
                        **asdict(job.config),
                        **job.green_details,
                        **score,
                        "generated_image": str(image_path),
                        "cellvit_geojson": str(job.cellvit_geojson),
                        "generation_seconds": generation_metadata.get(
                            "seconds", np.nan
                        ),
                    }
                    candidate_rows.append(row)
                    all_rows.append(row)

                best = max(candidate_rows, key=lambda row: float(row["spatial_score"]))
                selected_path = selected_dir / f"{stem}.png"
                copy_image(Path(best["generated_image"]), selected_path, True)
                selections.append(
                    {
                        "stem": stem,
                        "selected_image": str(selected_path),
                        **best,
                    }
                )
                if processed_inputs % args.save_every == 0 or processed_inputs == len(
                    stems
                ):
                    pd.DataFrame(all_rows).to_csv(
                        experiment_dir / "candidate_scores.csv", index=False
                    )
                    pd.DataFrame(selections).to_csv(
                        experiment_dir / "selected_candidates.csv", index=False
                    )
                print(
                    f"[rerank {processed_inputs}/{len(stems)}] {stem}: "
                    f"score={best['spatial_score']:.4f} {best['config_id']} "
                    f"seed={best['seed_index']}",
                    flush=True,
                )
    finally:
        phase_two.close()

    selected_metrics = None
    if not args.skip_metrics:
        selected_metrics = calculate_fid_kid(
            real_dir,
            selected_dir,
            args.kid_subset_size,
            args.kid_subsets,
            args.seed,
        )
        write_json(experiment_dir / "metrics_after_reranking.json", selected_metrics)
        print(f"[metrics after] {selected_metrics}", flush=True)

    comparison = {
        "before": baseline_metrics,
        "after": selected_metrics,
        "improvement_lower_is_better": (
            {
                "fid": baseline_metrics["fid"] - selected_metrics["fid"],
                "kid_mean": baseline_metrics["kid_mean"] - selected_metrics["kid_mean"],
            }
            if baseline_metrics is not None and selected_metrics is not None
            else None
        ),
    }
    write_json(experiment_dir / "fid_kid_comparison.json", comparison)
    print(f"Results written to {experiment_dir}")


if __name__ == "__main__":
    main()
