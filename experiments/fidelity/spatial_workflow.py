"""Shared baseline-generation path for both spatial fidelity experiments."""

from __future__ import annotations

from pathlib import Path

from .guidance import GenerationContext
from .workflow import ExperimentRuntime, deterministic_seed


def ensure_spatial_case(
    runtime: ExperimentRuntime, stem: str, base_seed: int
) -> tuple[Path, Path, dict]:
    sample = runtime.catalog.sample(stem)
    artifact_name = f"{stem}__spatial_baseline"
    seed = deterministic_seed(base_seed, stem, "spatial_fidelity")
    context = GenerationContext(
        stem=stem,
        condition_id="spatial_baseline",
        spatial_map=runtime.catalog.load_spatial(sample.spatial_path),
        morphology=sample.morphology,
        seed=seed,
        metadata={"experiment": "spatial_fidelity"},
    )
    image_path, generation_metadata = runtime.ensure_generated(context, artifact_name)
    geojson_path = runtime.ensure_cellvit(image_path, artifact_name)
    return image_path, geojson_path, generation_metadata
