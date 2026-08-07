"""Common CLI and resumable generation/CellViT workflow helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

from .cellvit import CellViTRunner, save_cellvit_geojson
from .data import DatasetCatalog
from .generation import PathOGenGenerator
from .guidance import GenerationContext, load_guidance_hook


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--cellvit-root", type=Path)
    parser.add_argument("--cellvit-model", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=25)
    parser.add_argument(
        "--stems",
        nargs="*",
        help="Optional explicit aligned case IDs; otherwise sample deterministically",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--spatial-strength", type=float, default=1.0)
    parser.add_argument(
        "--generator-precision", choices=("auto", "fp16", "fp32"), default="auto"
    )
    parser.add_argument(
        "--cellvit-precision", choices=("auto", "fp16", "fp32"), default="auto"
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=4,
        help="PathOGen inference batch size; CUDA OOM automatically halves it",
    )
    parser.add_argument(
        "--cellvit-batch-size",
        type=int,
        default=4,
        help="CellViT++ inference batch size; CUDA OOM automatically halves it",
    )
    parser.add_argument(
        "--guidance-hook",
        help="Optional module:factory returning experiments.fidelity.guidance.GuidanceHook",
    )
    parser.add_argument("--guidance-config", type=Path)
    parser.add_argument("--max-guidance-attempts", type=int, default=1)
    parser.add_argument(
        "--keep-rejected",
        action="store_true",
        help="Save the final candidate even if a guidance hook rejects every attempt",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Do not generate or infer; require all existing artifacts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and write the experiment plan without loading either model",
    )
    parser.add_argument("--overwrite", action="store_true")


def deterministic_seed(base_seed: int, *parts: str) -> int:
    payload = "|".join([str(base_seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_rgb_with_retry(
    path: Path, attempts: int = 6, initial_delay: float = 0.5
) -> Image.Image:
    """Fully read an RGB image, retrying transient mounted-Drive I/O failures."""
    last_error: OSError | None = None
    for attempt in range(1, attempts + 1):
        try:
            with Image.open(path) as image:
                image.load()
                return image.convert("RGB")
        except OSError as error:
            last_error = error
            if attempt == attempts:
                break
            delay = min(initial_delay * (2 ** (attempt - 1)), 8.0)
            print(
                f"[io] Read failed for {path} ({error}); retry "
                f"{attempt + 1}/{attempts} in {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)
    raise OSError(
        f"Could not read image after {attempts} attempts: {path}"
    ) from last_error


class ExperimentRuntime:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.catalog = DatasetCatalog(args.data_dir)
        self.output_dir = args.output_dir.resolve()
        self.generated_dir = self.output_dir / "generated"
        self.cellvit_dir = self.output_dir / "cellvit"
        self.metadata_dir = self.output_dir / "generation_metadata"
        self.hook = load_guidance_hook(args.guidance_hook, args.guidance_config)
        self._generator: PathOGenGenerator | None = None
        self._cellvit: CellViTRunner | None = None

    @property
    def generator(self) -> PathOGenGenerator:
        if self._generator is None:
            if self.args.checkpoint_dir is None:
                raise ValueError("--checkpoint-dir is required when generation is needed")
            self._generator = PathOGenGenerator(
                self.args.checkpoint_dir, precision=self.args.generator_precision
            )
        return self._generator

    @property
    def cellvit(self) -> CellViTRunner:
        if self._cellvit is None:
            if self.args.cellvit_root is None or self.args.cellvit_model is None:
                raise ValueError(
                    "--cellvit-root and --cellvit-model are required when inference is needed"
                )
            self._cellvit = CellViTRunner(
                self.args.cellvit_root,
                self.args.cellvit_model,
                precision=self.args.cellvit_precision,
            )
        return self._cellvit

    def selected_stems(self) -> list[str]:
        return self.catalog.select(self.args.num_images, self.args.seed, self.args.stems)

    def ensure_generated(
        self,
        context: GenerationContext,
        artifact_name: str,
        *,
        steps: int | None = None,
        spatial_strength: float | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        actual_steps = self.args.steps if steps is None else int(steps)
        actual_spatial_strength = (
            self.args.spatial_strength
            if spatial_strength is None
            else float(spatial_strength)
        )
        image_path = self.generated_dir / f"{safe_name(artifact_name)}.png"
        metadata_path = self.metadata_dir / f"{safe_name(artifact_name)}.json"
        if image_path.is_file() and not self.args.overwrite:
            metadata = (
                json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata_path.is_file()
                else {}
            )
            return image_path, metadata
        if self.args.analysis_only:
            raise FileNotFoundError(f"Required generated artifact missing: {image_path}")
        result = self.generator.generate(
            context,
            steps=actual_steps,
            spatial_strength=actual_spatial_strength,
            hook=self.hook,
            max_attempts=self.args.max_guidance_attempts,
        )
        if not result.decision.accept and not self.args.keep_rejected:
            raise RuntimeError(
                f"Guidance rejected {artifact_name} after {self.args.max_guidance_attempts} "
                f"attempts: {result.decision.reason}"
            )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        result.image.save(image_path)
        metadata = {
            "stem": result.context.stem,
            "condition_id": result.context.condition_id,
            "seed": result.context.seed,
            "attempt": result.context.attempt,
            "morphology": result.context.morphology.astype(float).tolist(),
            "steps": actual_steps,
            "spatial_strength": actual_spatial_strength,
            "seconds": round(result.seconds, 3),
            "accepted": result.decision.accept,
            "guidance_score": result.decision.score,
            "guidance_reason": result.decision.reason,
            "guidance_metadata": result.context.metadata,
            "generator": self.generator.describe(),
        }
        write_json(metadata_path, metadata)
        return image_path, metadata

    def ensure_generated_batch(
        self,
        contexts: list[GenerationContext],
        artifact_names: list[str],
        *,
        steps: int | None = None,
        spatial_strength: float | None = None,
    ) -> list[tuple[Path, dict[str, Any]]]:
        """Generate missing artifacts as one model batch and preserve input order."""
        if len(contexts) != len(artifact_names):
            raise ValueError("contexts and artifact_names must have equal length")
        actual_steps = self.args.steps if steps is None else int(steps)
        actual_spatial_strength = (
            self.args.spatial_strength
            if spatial_strength is None
            else float(spatial_strength)
        )
        outputs: list[tuple[Path, dict[str, Any]] | None] = [None] * len(contexts)
        missing_indices = []
        for index, name in enumerate(artifact_names):
            image_path = self.generated_dir / f"{safe_name(name)}.png"
            metadata_path = self.metadata_dir / f"{safe_name(name)}.json"
            if image_path.is_file() and not self.args.overwrite:
                metadata = (
                    json.loads(metadata_path.read_text(encoding="utf-8"))
                    if metadata_path.is_file()
                    else {}
                )
                outputs[index] = (image_path, metadata)
            else:
                missing_indices.append(index)
        if missing_indices:
            if self.args.analysis_only:
                first = self.generated_dir / f"{safe_name(artifact_names[missing_indices[0]])}.png"
                raise FileNotFoundError(f"Required generated artifact missing: {first}")
            results = self.generator.generate_batch(
                [contexts[index] for index in missing_indices],
                steps=actual_steps,
                spatial_strength=actual_spatial_strength,
                hook=self.hook,
                max_attempts=self.args.max_guidance_attempts,
            )
            for index, result in zip(missing_indices, results):
                name = artifact_names[index]
                if not result.decision.accept and not self.args.keep_rejected:
                    raise RuntimeError(
                        f"Guidance rejected {name}: {result.decision.reason}"
                    )
                image_path = self.generated_dir / f"{safe_name(name)}.png"
                metadata_path = self.metadata_dir / f"{safe_name(name)}.json"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                result.image.save(image_path)
                metadata = {
                    "stem": result.context.stem,
                    "condition_id": result.context.condition_id,
                    "seed": result.context.seed,
                    "attempt": result.context.attempt,
                    "morphology": result.context.morphology.astype(float).tolist(),
                    "steps": actual_steps,
                    "spatial_strength": actual_spatial_strength,
                    "seconds": round(result.seconds, 3),
                    "accepted": result.decision.accept,
                    "guidance_score": result.decision.score,
                    "guidance_reason": result.decision.reason,
                    "guidance_metadata": result.context.metadata,
                    "generator": self.generator.describe(),
                }
                write_json(metadata_path, metadata)
                outputs[index] = (image_path, metadata)
        if any(output is None for output in outputs):
            raise RuntimeError("Internal error: incomplete batched generation outputs")
        return [output for output in outputs if output is not None]

    def ensure_cellvit(self, image_path: Path, artifact_name: str) -> Path:
        geojson_path = self.cellvit_dir / f"{safe_name(artifact_name)}.geojson"
        if geojson_path.is_file() and not self.args.overwrite:
            return geojson_path
        if self.args.analysis_only:
            raise FileNotFoundError(f"Required CellViT artifact missing: {geojson_path}")
        cells = self.cellvit.infer(load_rgb_with_retry(image_path))
        save_cellvit_geojson(cells, geojson_path)
        return geojson_path

    def ensure_cellvit_batch(
        self, image_paths: list[Path], artifact_names: list[str]
    ) -> list[Path]:
        """Segment missing image artifacts as one CellViT++ model batch."""
        if len(image_paths) != len(artifact_names):
            raise ValueError("image_paths and artifact_names must have equal length")
        outputs: list[Path | None] = [None] * len(image_paths)
        missing_indices = []
        for index, name in enumerate(artifact_names):
            geojson_path = self.cellvit_dir / f"{safe_name(name)}.geojson"
            if geojson_path.is_file() and not self.args.overwrite:
                outputs[index] = geojson_path
            else:
                missing_indices.append(index)
        if missing_indices:
            if self.args.analysis_only:
                first = self.cellvit_dir / f"{safe_name(artifact_names[missing_indices[0]])}.geojson"
                raise FileNotFoundError(f"Required CellViT artifact missing: {first}")
            images = []
            for index in missing_indices:
                images.append(load_rgb_with_retry(image_paths[index]))
            cell_batches = self.cellvit.infer_batch(images)
            for index, cells in zip(missing_indices, cell_batches):
                geojson_path = self.cellvit_dir / f"{safe_name(artifact_names[index])}.geojson"
                save_cellvit_geojson(cells, geojson_path)
                outputs[index] = geojson_path
        if any(output is None for output in outputs):
            raise RuntimeError("Internal error: incomplete batched CellViT outputs")
        return [output for output in outputs if output is not None]

    def close(self) -> None:
        if self._generator is not None:
            self._generator.unload()
        if self._cellvit is not None:
            self._cellvit.unload()
