#!/usr/bin/env python
"""Generate the paper's spatial- and morphology-fidelity tables end to end."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.colab.layout import DEFAULT_CONFIG, RuntimePaths
from experiments.fidelity.constants import MORPH_FEATURES
from experiments.fidelity.data import DatasetCatalog, load_cells
from experiments.fidelity.evaluators import (
    ensure_cellvit_predictions,
    ensure_hovernet_predictions,
    ensure_stardist_predictions,
)
from experiments.fidelity.guidance import GenerationContext
from experiments.fidelity.measurements import morphology_measurements
from experiments.fidelity.table_metrics import (
    TABLE_MORPH_FEATURES,
    patient_id,
    summarize_across_morphology,
    summarize_controlled_morphology,
    summarize_spatial,
)
from experiments.fidelity.workflow import (
    ExperimentRuntime,
    load_rgb_with_retry,
    path_is_file_with_retry,
    write_json,
)

DEFAULT_RERANK_DIR = Path(
    "/content/drive/MyDrive/PathOGenResults/cellvit_rerank_fid_kid/"
    "cellvit_rerank_6e25c52f8dcb"
)
DEFAULT_DOSES = (-1.0, -0.5, 0.0, 0.5, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--cellvit-root", type=Path)
    parser.add_argument("--cellvit-model", type=Path)
    parser.add_argument("--rerank-dir", type=Path, default=DEFAULT_RERANK_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--scratch-dir", type=Path, default=Path("/content/pathogen_fidelity")
    )
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--controlled-images", type=int, default=200)
    parser.add_argument(
        "--controlled-levels", type=float, nargs="+", default=DEFAULT_DOSES
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--spatial-strength", type=float, default=2.0)
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--cellvit-batch-size", type=int, default=32)
    parser.add_argument("--hovernet-batch-size", type=int, default=32)
    parser.add_argument(
        "--hovernet-memory-fraction",
        type=float,
        default=0.8,
        help="Maximum GPU-memory fraction exposed to official HoVer-Net inference",
    )
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
        "--evaluators",
        nargs="+",
        choices=("cellvit", "hovernet", "stardist"),
        default=("cellvit", "hovernet", "stardist"),
    )
    parser.add_argument("--hovernet-root", type=Path)
    parser.add_argument("--hovernet-model", type=Path)
    parser.add_argument(
        "--hovernet-predictions-dir",
        type=Path,
        help="Optional existing official HoVer-Net JSON output tree",
    )
    parser.add_argument(
        "--hovernet-model-mode", choices=("fast", "original"), default="fast"
    )
    parser.add_argument("--stardist-model", default="2D_versatile_he")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    if config_path.is_file():
        paths = RuntimePaths.read(config_path)
        args.data_dir = args.data_dir or paths.data_dir
        args.checkpoint_dir = args.checkpoint_dir or paths.checkpoint_dir
        args.cellvit_root = args.cellvit_root or paths.cellvit_root
        args.cellvit_model = args.cellvit_model or paths.cellvit_model
        if args.output_dir is None:
            args.output_dir = paths.output_root / "paper_fidelity_tables"
    if args.output_dir is None:
        args.output_dir = args.rerank_dir.parent.parent / "paper_fidelity_tables"
    if args.data_dir is None:
        raise ValueError(
            "Missing --data-dir; run experiments/colab/setup_colab.py first"
        )
    if args.checkpoint_dir is None and not args.dry_run:
        raise ValueError("Missing --checkpoint-dir required for controlled generation")
    if (
        "cellvit" in args.evaluators
        and not args.dry_run
        and (args.cellvit_root is None or args.cellvit_model is None)
    ):
        raise ValueError("CellViT++ requires --cellvit-root and --cellvit-model")
    if (
        "hovernet" in args.evaluators
        and not args.dry_run
        and args.hovernet_predictions_dir is None
        and (args.hovernet_root is None or args.hovernet_model is None)
    ):
        raise ValueError(
            "HoVer-Net requires --hovernet-root/--hovernet-model or "
            "--hovernet-predictions-dir"
        )

    # Attributes consumed by ExperimentRuntime. Guidance is deliberately disabled:
    # controlled rows must isolate one morphology coordinate and one fixed noise seed.
    args.guidance_hook = None
    args.guidance_config = None
    args.max_guidance_attempts = 1
    args.keep_rejected = False
    args.analysis_only = False
    return args


def _resolve_saved_path(raw: object, candidates: list[Path]) -> Path | None:
    if raw is not None and not pd.isna(raw):
        path = Path(str(raw)).expanduser()
        if path_is_file_with_retry(path):
            return path.absolute()
    for candidate in candidates:
        if path_is_file_with_retry(candidate):
            return candidate.absolute()
        if raw is not None and not pd.isna(raw):
            joined = candidate / Path(str(raw)).name
            if path_is_file_with_retry(joined):
                return joined.absolute()
    return None


def read_csv_with_retry(path: Path, attempts: int = 6) -> pd.DataFrame:
    last_error: OSError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return pd.read_csv(path)
        except OSError as error:
            last_error = error
            if attempt == attempts:
                break
            delay = min(0.5 * (2 ** (attempt - 1)), 8.0)
            print(
                f"[io] CSV read failed for {path} ({error}); retry "
                f"{attempt + 1}/{attempts} in {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)
    raise OSError(
        f"Could not read CSV after {attempts} attempts: {path}"
    ) from last_error


def load_selected_cases(
    rerank_dir: Path, catalog: DatasetCatalog, num_images: int
) -> pd.DataFrame:
    rerank_dir = rerank_dir.expanduser().resolve()
    selections_path = rerank_dir / "selected_candidates.csv"
    if not path_is_file_with_retry(selections_path):
        raise FileNotFoundError(f"Reranking selections missing: {selections_path}")
    selections = read_csv_with_retry(selections_path)
    if "stem" not in selections or "seed" not in selections:
        raise ValueError("selected_candidates.csv must contain stem and seed columns")
    selections["stem"] = selections["stem"].astype(str)
    selections = selections.drop_duplicates("stem", keep="last")
    rows = []
    for record in selections.to_dict(orient="records"):
        stem = record["stem"]
        if stem not in catalog.stems:
            continue
        image_path = _resolve_saved_path(
            record.get("selected_image"),
            [rerank_dir / "metric_sets" / "selected" / f"{stem}.png"],
        )
        if image_path is None:
            continue
        cellvit_path = _resolve_saved_path(
            record.get("cellvit_geojson"),
            [rerank_dir / "cellvit"],
        )
        selected_green = record.get("green_applied")
        if selected_green is None or pd.isna(selected_green):
            selected_green = catalog.morphology.loc[stem, "g_mean"]
        rows.append(
            {
                "stem": stem,
                "selected_image": str(image_path),
                "selected_seed": int(record["seed"]),
                "selected_config": record.get("config_id", "unknown"),
                "selected_green_applied": float(selected_green),
                "existing_cellvit": str(cellvit_path) if cellvit_path else None,
            }
        )
        if len(rows) == num_images:
            break
    if len(rows) != num_images:
        raise RuntimeError(
            f"Found {len(rows)} complete selected cases in {rerank_dir}; "
            f"requested {num_images}"
        )
    return pd.DataFrame(rows)


def random_tile_derangement(stems: list[str], seed: int) -> dict[str, str]:
    """Return a fixed one-to-one random real-tile pairing from another patient."""
    rng = random.Random(seed)
    candidates = list(stems)
    for _ in range(1000):
        rng.shuffle(candidates)
        if all(
            patient_id(source) != patient_id(target)
            for source, target in zip(stems, candidates)
        ):
            return dict(zip(stems, candidates))
    # Deterministic fallback for unusually patient-concentrated subsets.
    ordered = sorted(stems, key=lambda stem: (patient_id(stem), stem))
    for shift in range(1, len(ordered)):
        rotated = ordered[shift:] + ordered[:shift]
        if all(
            patient_id(source) != patient_id(target)
            for source, target in zip(ordered, rotated)
        ):
            mapping = dict(zip(ordered, rotated))
            return {stem: mapping[stem] for stem in stems}
    raise RuntimeError(
        "Could not construct a random-tile pairing across different patients"
    )


def build_controlled_plan(
    selected: pd.DataFrame,
    catalog: DatasetCatalog,
    count: int,
    levels: list[float],
    seed: int,
) -> pd.DataFrame:
    if sorted(levels) != list(levels) or len(set(levels)) < 4 or 0.0 not in levels:
        raise ValueError(
            "Controlled levels must be sorted, unique, include zero, and contain >=4 doses"
        )
    selected_by_stem = selected.set_index("stem")
    controlled_features = list(TABLE_MORPH_FEATURES.values())
    shuffled = selected["stem"].astype(str).tolist()
    random.Random(seed).shuffle(shuffled)
    eligible = [
        stem
        for stem in shuffled
        if np.isfinite(
            catalog.morphology.loc[stem, controlled_features].to_numpy(dtype=float)
        ).all()
    ]
    if len(eligible) < count:
        raise RuntimeError(
            f"Only {len(eligible)} selected cases have finite values for every "
            f"controlled morphology coordinate; requested {count}."
        )

    # The morphology parquet was standardized on the full training corpus before
    # the validation split was created. One SD is therefore exactly one coordinate
    # unit; validation-subset statistics must not redefine or clip that unit.
    controlled_stems = eligible[:count]
    rows = []
    for feature in controlled_features:
        for stem in controlled_stems:
            baseline = float(catalog.morphology.loc[stem, feature])
            fixed_seed = int(selected_by_stem.loc[stem, "selected_seed"])
            for level in levels:
                target = baseline + float(level)
                token = (
                    f"{level:+.2f}".replace("+", "p")
                    .replace("-", "m")
                    .replace(".", "p")
                )
                artifact_id = (
                    f"{stem}__controlled__neutral"
                    if level == 0
                    else f"{stem}__controlled__{feature}__{token}sd"
                )
                rows.append(
                    {
                        "plan_id": f"{stem}__{feature}__{token}sd",
                        "stem": stem,
                        "patient": patient_id(stem),
                        "feature": feature,
                        "dose_sd": float(level),
                        "baseline_value": baseline,
                        "feature_sd": 1.0,
                        "input_delta_std": float(level),
                        "input_value": target,
                        "seed": fixed_seed,
                        "artifact_id": artifact_id,
                    }
                )
    return pd.DataFrame(rows)


def generate_controlled_images(
    args: argparse.Namespace,
    catalog: DatasetCatalog,
    plan: pd.DataFrame,
) -> dict[str, Path]:
    # Keep only scalar/path records between batches. A materialized context owns a
    # 512x512x5 map, so retaining thousands of contexts would consume many GiB of RAM.
    records = plan.drop_duplicates("artifact_id", keep="first").to_dict(
        orient="records"
    )
    runtime = ExperimentRuntime(args)
    outputs: dict[str, Path] = {}
    try:
        for start in range(0, len(records), args.generation_batch_size):
            group = records[start : start + args.generation_batch_size]
            contexts = []
            artifact_ids = []
            for record in group:
                artifact_id = str(record["artifact_id"])
                stem = str(record["stem"])
                sample = catalog.sample(stem)
                morphology = sample.morphology.copy()
                feature = str(record["feature"])
                if float(record["dose_sd"]) != 0.0:
                    morphology[MORPH_FEATURES.index(feature)] = float(
                        record["input_value"]
                    )
                contexts.append(
                    GenerationContext(
                        stem=stem,
                        condition_id=artifact_id,
                        spatial_map=catalog.load_spatial(sample.spatial_path),
                        morphology=morphology,
                        seed=int(record["seed"]),
                        metadata={
                            "experiment": "paper_morphology_controlled",
                            "feature": feature,
                            "dose_sd": float(record["dose_sd"]),
                            "fixed_seed_across_doses": True,
                        },
                    )
                )
                artifact_ids.append(artifact_id)
            generated = runtime.ensure_generated_batch(
                contexts,
                artifact_ids,
                steps=args.steps,
                spatial_strength=args.spatial_strength,
            )
            for artifact_id, (path, _) in zip(artifact_ids, generated):
                outputs[artifact_id] = path
            print(
                f"[controlled generation] "
                f"{min(start + len(group), len(records))}/{len(records)}",
                flush=True,
            )
    finally:
        runtime.close()
    return outputs


def ensure_measurements(
    evaluator: str,
    cohort: str,
    images: dict[str, Path],
    predictions: dict[str, Path],
    catalog: DatasetCatalog,
    destination: Path,
    controlled_plan: pd.DataFrame | None,
    across_inputs: dict[str, np.ndarray] | None,
    save_every: int,
    overwrite: bool,
) -> pd.DataFrame:
    if path_is_file_with_retry(destination) and not overwrite:
        existing = read_csv_with_retry(destination)
    else:
        existing = pd.DataFrame()
    key_column = "plan_id" if controlled_plan is not None else "artifact_id"
    valid_keys = (
        set(controlled_plan["plan_id"].astype(str))
        if controlled_plan is not None
        else set(images)
    )
    if key_column in existing:
        existing = existing[existing[key_column].astype(str).isin(valid_keys)].copy()
    completed = (
        set(existing[key_column].astype(str)) if key_column in existing else set()
    )
    rows = existing.to_dict(orient="records")
    if controlled_plan is None:
        pending = [
            {"plan_id": artifact_id, "artifact_id": artifact_id}
            for artifact_id in images
        ]
    else:
        pending = controlled_plan.to_dict(orient="records")
    pending = [record for record in pending if str(record[key_column]) not in completed]
    measurement_cache: dict[str, dict[str, float]] = {}
    for index, record in enumerate(pending, start=1):
        artifact_id = str(record["artifact_id"])
        if artifact_id not in measurement_cache:
            image = load_rgb_with_retry(images[artifact_id])
            measurement_cache[artifact_id] = morphology_measurements(
                image, load_cells(predictions[artifact_id])
            )
        measured = measurement_cache[artifact_id]
        if controlled_plan is None:
            stem = artifact_id
            row: dict[str, Any] = {
                "evaluator": evaluator,
                "cohort": cohort,
                "artifact_id": artifact_id,
                "stem": stem,
                "patient": patient_id(stem),
                "image_path": str(images[artifact_id]),
                "prediction_path": str(predictions[artifact_id]),
            }
            morphology = (
                across_inputs[stem]
                if across_inputs is not None
                else catalog.sample(stem).morphology
            )
            row.update(
                {
                    f"input_{name}": float(value)
                    for name, value in zip(MORPH_FEATURES, morphology)
                }
            )
            row.update({f"measured_{name}": value for name, value in measured.items()})
        else:
            row = {
                "evaluator": evaluator,
                "cohort": cohort,
                "plan_id": record["plan_id"],
                "artifact_id": artifact_id,
                "stem": record["stem"],
                "patient": record["patient"],
                "image_path": str(images[artifact_id]),
                "prediction_path": str(predictions[artifact_id]),
                "feature": record["feature"],
                "dose_sd": float(record["dose_sd"]),
                "input_value": float(record["input_value"]),
                "measured_value": float(measured[record["feature"]]),
            }
        rows.append(row)
        if index % save_every == 0 or index == len(pending):
            destination.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(destination, index=False)
            print(
                f"[{evaluator} {cohort} measurements] "
                f"{len(completed) + index}/{len(completed) + len(pending)}",
                flush=True,
            )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns[1:]:
        display[column] = display[column].map(
            lambda value: (
                "N/A" if value is None or pd.isna(value) else f"{float(value):.3f}"
            )
        )
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in display.to_numpy()]
    return "\n".join([header, separator, *rows])


def main() -> None:
    args = parse_args()
    if args.num_images < 3 or args.controlled_images < 3:
        raise ValueError("Use at least three across-image and controlled source tiles")
    if args.bootstrap < 0 or args.save_every < 1:
        raise ValueError("bootstrap must be non-negative and save-every positive")
    if not 0 < args.hovernet_memory_fraction <= 1:
        raise ValueError("hovernet-memory-fraction must be in (0, 1]")

    args.data_dir = args.data_dir.expanduser().resolve()
    args.rerank_dir = args.rerank_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.scratch_dir = args.scratch_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    catalog = DatasetCatalog(args.data_dir)
    selected = load_selected_cases(args.rerank_dir, catalog, args.num_images)
    random_mapping = random_tile_derangement(selected["stem"].tolist(), args.seed + 900)
    selected["random_tile_stem"] = selected["stem"].map(random_mapping)
    selected["random_tile_image"] = selected["random_tile_stem"].map(
        lambda stem: str(catalog.sample(stem).image_path)
    )
    selected.to_csv(args.output_dir / "spatial_case_plan.csv", index=False)

    controlled_plan = build_controlled_plan(
        selected,
        catalog,
        args.controlled_images,
        list(args.controlled_levels),
        args.seed,
    )
    generation_suffix = (
        f"__c{args.spatial_strength:g}".replace(".", "p") + f"__s{args.steps}"
    )
    controlled_plan["artifact_id"] = (
        controlled_plan["artifact_id"].astype(str) + generation_suffix
    )
    controlled_plan["plan_id"] = (
        controlled_plan["plan_id"].astype(str) + generation_suffix
    )
    controlled_plan.to_csv(
        args.output_dir / "controlled_condition_plan.csv", index=False
    )
    write_json(
        args.output_dir / "experiment_manifest.json",
        {
            "experiment": "paper_spatial_and_morphology_fidelity_tables",
            "rerank_dir": str(args.rerank_dir),
            "data_dir": str(args.data_dir),
            "num_across_images": args.num_images,
            "num_controlled_images_per_feature": args.controlled_images,
            "controlled_plan_rows": len(controlled_plan),
            "controlled_unique_images": int(controlled_plan["artifact_id"].nunique()),
            "controlled_levels_sd": list(args.controlled_levels),
            "controlled_intervention_space": (
                "full-training-corpus standardized morphology coordinates"
            ),
            "features": list(TABLE_MORPH_FEATURES.values()),
            "evaluators": list(args.evaluators),
            "generator_memory_mode": args.generator_memory_mode,
            "hovernet_memory_fraction": args.hovernet_memory_fraction,
            "random_tile": "fixed one-to-one real-tile pairing from another patient; source GeoJSON reused",
            "centroid_matching": {
                "radii_pixels": [25, 50],
                "assignment": "Hungarian one-to-one",
                "typed_for": ["CellViT++", "HoVer-Net", "Random Tile"],
                "untyped_for": ["StarDist"],
            },
            "controlled_generation": {
                "steps": args.steps,
                "spatial_strength": args.spatial_strength,
                "same_seed_within_source": True,
                "neutral_morphology": True,
                "one_changed_coordinate": True,
                "standardized_unit_per_sd": 1.0,
                "validation_subset_clipping": False,
            },
            "bootstrap": args.bootstrap,
            "bootstrap_unit": "TCGA patient inferred from tile stem",
            "seed": args.seed,
        },
    )
    if args.dry_run:
        print(
            f"Dry run passed: {len(selected)} selected tiles; {len(controlled_plan)} "
            "controlled plan rows; no models loaded.",
            flush=True,
        )
        return

    selected_images = {
        row.stem: Path(row.selected_image) for row in selected.itertuples(index=False)
    }
    selected_input_vectors = {}
    green_index = MORPH_FEATURES.index("g_mean")
    for row in selected.itertuples(index=False):
        vector = catalog.sample(row.stem).morphology.copy()
        vector[green_index] = float(row.selected_green_applied)
        selected_input_vectors[row.stem] = vector
    controlled_images = generate_controlled_images(args, catalog, controlled_plan)
    all_prediction_sets: dict[str, tuple[dict[str, Path], dict[str, Path]]] = {}
    prediction_root = args.output_dir / "predictions"

    if "cellvit" in args.evaluators:
        reusable = {
            row.stem: Path(row.existing_cellvit)
            for row in selected.itertuples(index=False)
            if isinstance(row.existing_cellvit, str) and row.existing_cellvit
        }
        selected_predictions = ensure_cellvit_predictions(
            selected_images,
            prediction_root / "cellvit" / "selected",
            args.cellvit_root,
            args.cellvit_model,
            args.cellvit_batch_size,
            args.cellvit_precision,
            existing=reusable,
            overwrite=args.overwrite,
        )
        controlled_predictions = ensure_cellvit_predictions(
            controlled_images,
            prediction_root / "cellvit" / "controlled",
            args.cellvit_root,
            args.cellvit_model,
            args.cellvit_batch_size,
            args.cellvit_precision,
            overwrite=args.overwrite,
        )
        all_prediction_sets["CellViT++"] = (
            selected_predictions,
            controlled_predictions,
        )

    if "hovernet" in args.evaluators:
        selected_predictions = ensure_hovernet_predictions(
            selected_images,
            prediction_root / "hovernet" / "selected",
            args.scratch_dir / "hovernet_selected",
            args.hovernet_root,
            args.hovernet_model,
            args.hovernet_predictions_dir,
            args.hovernet_batch_size,
            args.hovernet_model_mode,
            args.overwrite,
            memory_fraction=args.hovernet_memory_fraction,
        )
        controlled_predictions = ensure_hovernet_predictions(
            controlled_images,
            prediction_root / "hovernet" / "controlled",
            args.scratch_dir / "hovernet_controlled",
            args.hovernet_root,
            args.hovernet_model,
            args.hovernet_predictions_dir,
            args.hovernet_batch_size,
            args.hovernet_model_mode,
            args.overwrite,
            memory_fraction=args.hovernet_memory_fraction,
        )
        all_prediction_sets["HoVer-Net"] = (
            selected_predictions,
            controlled_predictions,
        )

    if "stardist" in args.evaluators:
        selected_predictions = ensure_stardist_predictions(
            selected_images,
            prediction_root / "stardist" / "selected",
            args.stardist_model,
            args.overwrite,
        )
        controlled_predictions = ensure_stardist_predictions(
            controlled_images,
            prediction_root / "stardist" / "controlled",
            args.stardist_model,
            args.overwrite,
        )
        all_prediction_sets["StarDist"] = (selected_predictions, controlled_predictions)

    spatial_rows = []
    spatial_cases = []
    per_type_rows = []
    spatial_details = []
    row_seed = args.seed + 1000
    for evaluator, (selected_predictions, _) in all_prediction_sets.items():
        pairs = [
            (
                stem,
                load_cells(catalog.sample(stem).geojson_path),
                load_cells(selected_predictions[stem]),
            )
            for stem in selected["stem"]
        ]
        row, cases, types, detail = summarize_spatial(
            pairs,
            evaluator,
            typed=evaluator != "StarDist",
            bootstrap=args.bootstrap,
            seed=row_seed + len(spatial_rows) * 100,
        )
        spatial_rows.append(row)
        spatial_cases.append(cases)
        per_type_rows.append(types)
        spatial_details.append(detail)

    random_pairs = [
        (
            row.stem,
            load_cells(catalog.sample(row.stem).geojson_path),
            load_cells(catalog.sample(row.random_tile_stem).geojson_path),
        )
        for row in selected.itertuples(index=False)
    ]
    row, cases, types, detail = summarize_spatial(
        random_pairs,
        "Random Tile",
        typed=True,
        bootstrap=args.bootstrap,
        seed=row_seed + 900,
    )
    spatial_rows.append(row)
    spatial_cases.append(cases)
    per_type_rows.append(types)
    spatial_details.append(detail)
    spatial_table = pd.DataFrame(spatial_rows)
    desired_order = ["CellViT++", "HoVer-Net", "StarDist", "Random Tile"]
    spatial_table["_order"] = spatial_table["Method"].map(
        {name: index for index, name in enumerate(desired_order)}
    )
    spatial_table = spatial_table.sort_values("_order").drop(columns="_order")
    spatial_table.to_csv(args.output_dir / "T1_spatial_fidelity.csv", index=False)
    pd.concat(spatial_cases, ignore_index=True).to_csv(
        args.output_dir / "spatial_per_tile_metrics.csv", index=False
    )
    pd.concat(per_type_rows, ignore_index=True).to_csv(
        args.output_dir / "spatial_per_type_count_correlations.csv", index=False
    )
    pd.DataFrame(spatial_details).to_csv(
        args.output_dir / "spatial_confidence_intervals.csv", index=False
    )

    morphology_rows = []
    morphology_details = []
    controlled_source_frames = []
    for evaluator_index, (
        evaluator,
        (selected_predictions, controlled_predictions),
    ) in enumerate(all_prediction_sets.items()):
        across = ensure_measurements(
            evaluator,
            "selected",
            selected_images,
            selected_predictions,
            catalog,
            args.output_dir
            / "measurements"
            / f"{evaluator.lower().replace('-', '')}_across.csv",
            None,
            selected_input_vectors,
            args.save_every,
            args.overwrite,
        )
        controlled = ensure_measurements(
            evaluator,
            "controlled",
            controlled_images,
            controlled_predictions,
            catalog,
            args.output_dir
            / "measurements"
            / f"{evaluator.lower().replace('-', '')}_controlled.csv",
            controlled_plan,
            None,
            args.save_every,
            args.overwrite,
        )
        across_name = f"Across Images {evaluator}"
        controlled_name = f"Within Image Controlled {evaluator}"
        across_row, across_details = summarize_across_morphology(
            across, across_name, args.bootstrap, args.seed + evaluator_index * 100
        )
        controlled_row, controlled_details, source_frame = (
            summarize_controlled_morphology(
                controlled,
                controlled_name,
                args.bootstrap,
                args.seed + 500 + evaluator_index * 100,
            )
        )
        morphology_rows.extend([across_row, controlled_row])
        morphology_details.extend(across_details + controlled_details)
        controlled_source_frames.append(source_frame)

    morphology_table = pd.DataFrame(morphology_rows)
    morphology_order = [
        "Across Images CellViT++",
        "Across Images HoVer-Net",
        "Across Images StarDist",
        "Within Image Controlled CellViT++",
        "Within Image Controlled HoVer-Net",
        "Within Image Controlled StarDist",
    ]
    morphology_table["_order"] = morphology_table["Method"].map(
        {name: index for index, name in enumerate(morphology_order)}
    )
    morphology_table = morphology_table.sort_values("_order").drop(columns="_order")
    morphology_table.to_csv(args.output_dir / "T2_morphology_fidelity.csv", index=False)
    pd.DataFrame(morphology_details).to_csv(
        args.output_dir / "morphology_confidence_intervals.csv", index=False
    )
    pd.concat(controlled_source_frames, ignore_index=True).to_csv(
        args.output_dir / "morphology_controlled_per_source_rho.csv", index=False
    )

    tables_md = "\n".join(
        [
            "# PathOGen fidelity tables",
            "",
            "## T1. Spatial Fidelity Check",
            "",
            markdown_table(spatial_table),
            "",
            "Per-type count is the macro-average across neoplastic/tumor, inflammatory/immune, connective/stromal, dead, and non-neoplastic epithelial nuclei. StarDist is segmentation-only, so its typed count is N/A. Centroid matching is one-to-one Hungarian matching; it is type-aware for CellViT++, HoVer-Net, and Random Tile, and position-only for StarDist.",
            "",
            "## T2. Morphology Fidelity Check",
            "",
            markdown_table(morphology_table),
            "",
            "Across-image entries are Spearman correlations over the selected reranked images. Controlled entries are the median within-tile Spearman correlation across −1, −0.5, 0, +0.5, and +1 SD with a fixed noise seed and all other controls held constant. Macro ρ is the unweighted mean of the seven displayed features.",
            "",
            "CellViT++ was used for candidate selection and is therefore an in-loop evaluator; HoVer-Net and StarDist are out-of-loop checks.",
        ]
    )
    (args.output_dir / "TABLES.md").write_text(tables_md, encoding="utf-8")
    print(tables_md, flush=True)
    print(f"\nAll table artifacts written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
