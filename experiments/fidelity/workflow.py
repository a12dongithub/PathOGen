"""Common CLI and resumable generation/CellViT workflow helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    ) -> tuple[Path, dict[str, Any]]:
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
            steps=self.args.steps,
            spatial_strength=self.args.spatial_strength,
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
            "steps": self.args.steps,
            "spatial_strength": self.args.spatial_strength,
            "seconds": round(result.seconds, 3),
            "accepted": result.decision.accept,
            "guidance_score": result.decision.score,
            "guidance_reason": result.decision.reason,
            "guidance_metadata": result.context.metadata,
            "generator": self.generator.describe(),
        }
        write_json(metadata_path, metadata)
        return image_path, metadata

    def ensure_cellvit(self, image_path: Path, artifact_name: str) -> Path:
        geojson_path = self.cellvit_dir / f"{safe_name(artifact_name)}.geojson"
        if geojson_path.is_file() and not self.args.overwrite:
            return geojson_path
        if self.args.analysis_only:
            raise FileNotFoundError(f"Required CellViT artifact missing: {geojson_path}")
        cells = self.cellvit.infer(Image.open(image_path).convert("RGB"))
        save_cellvit_geojson(cells, geojson_path)
        return geojson_path

    def close(self) -> None:
        if self._generator is not None:
            self._generator.unload()
