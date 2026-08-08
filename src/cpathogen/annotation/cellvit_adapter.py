"""Tile-oriented CellViT++ adapter for source and counterfactual H&E images.

The model classes and CPU HoVer-Net postprocessor come from the pinned upstream
snapshot under ``third_party/cellvit_plus_plus``. This adapter avoids the
upstream WSI runner's CUDA/CuPy/Ray requirement and preserves pair metadata for
counterfactual validation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .geojson import (
    NucleusPrediction,
    PANNUKE_CLASS_NAMES,
    load_and_validate_geojson,
    predictions_to_geojson,
    validate_geojson,
    write_geojson,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CELLVIT_ROOT = REPOSITORY_ROOT / "third_party/cellvit_plus_plus"
DEFAULT_MODEL = (
    REPOSITORY_ROOT
    / "models/cellvit_plus_plus/cellvit_sam_h_x40_amp_001/model.pth"
)
DEFAULT_INPUT_DIR = REPOSITORY_ROOT / "data/images"
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "data/geojsons"
)
UPSTREAM_COMMIT = "463c5c44bfdebfbe3943597eaa84daf3f5e26a5f"
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageTask:
    image_path: Path
    output_path: Path
    source_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelInfo:
    architecture: str
    backbone: str
    num_nuclei_classes: int
    num_tissue_classes: int
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]
    maximum_input_size: int
    patch_size: int


def _unflatten_dict(values: dict[str, Any], separator: str = ".") -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in values.items():
        target = output
        parts = str(key).split(separator)
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "auto":
        return torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtypes[requested]
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CellViT++ CPU inference requires float32")
    return dtype


class CellViTTileAnnotator:
    """Load CellViT-SAM-H once and annotate prepared images one at a time."""

    def __init__(
        self,
        model_path: str | Path,
        cellvit_root: str | Path,
        *,
        device: str = "auto",
        dtype: str = "auto",
        magnification: int = 40,
        min_type_probability: float = 0.0,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.cellvit_root = Path(cellvit_root).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"CellViT++ model not found: {self.model_path}")
        if not (self.cellvit_root / "cellvit").is_dir():
            raise FileNotFoundError(
                f"Pinned CellViT++ source not found under: {self.cellvit_root}"
            )
        if not 0.0 <= min_type_probability <= 1.0:
            raise ValueError("min_type_probability must be in [0, 1]")
        if magnification not in (20, 40):
            raise ValueError("magnification must be 20 or 40")
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype, self.device)
        self.magnification = magnification
        self.min_type_probability = min_type_probability
        self.model, self.model_info = self._load_model()
        self.postprocessor = self._load_postprocessor()

    def _load_model(self) -> tuple[torch.nn.Module, ModelInfo]:
        if str(self.cellvit_root) not in sys.path:
            sys.path.insert(0, str(self.cellvit_root))
        from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM

        checkpoint = torch.load(
            self.model_path, map_location="cpu", weights_only=False
        )
        architecture = str(checkpoint.get("arch"))
        if architecture != "CellViTSAM":
            raise ValueError(
                f"This tile adapter currently supports CellViTSAM, found {architecture}"
            )
        config = _unflatten_dict(checkpoint["config"])
        backbone = str(config["model"]["backbone"])
        model = CellViTSAM(
            model_path=None,
            num_nuclei_classes=int(config["data"]["num_nuclei_classes"]),
            num_tissue_classes=int(config["data"]["num_tissue_classes"]),
            vit_structure=backbone,
            regression_loss=bool(config["model"].get("regression_loss", False)),
        )
        load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(f"CellViT++ checkpoint mismatch: {load_result}")
        normalize = config.get("transformations", {}).get("normalize", {})
        mean = tuple(float(value) for value in normalize.get("mean", (0.5, 0.5, 0.5)))
        std = tuple(float(value) for value in normalize.get("std", (0.5, 0.5, 0.5)))
        patch_size = int(model.patch_size)
        maximum_input_size = int(model.encoder.pos_embed.shape[1] * patch_size)
        info = ModelInfo(
            architecture=architecture,
            backbone=backbone,
            num_nuclei_classes=int(config["data"]["num_nuclei_classes"]),
            num_tissue_classes=int(config["data"]["num_tissue_classes"]),
            normalization_mean=mean,
            normalization_std=std,
            maximum_input_size=maximum_input_size,
            patch_size=patch_size,
        )
        del checkpoint
        gc.collect()
        model.to(device=self.device, dtype=self.dtype).eval().requires_grad_(False)
        return model, info

    def _load_postprocessor(self):
        from cellvit.models.cell_segmentation.postprocessing import (
            DetectionCellPostProcessor,
        )

        return DetectionCellPostProcessor(
            nr_types=self.model_info.num_nuclei_classes,
            magnification=self.magnification,
        )

    def _prepare_image(self, path: Path) -> tuple[torch.Tensor, int, int]:
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            if (
                width > self.model_info.maximum_input_size
                or height > self.model_info.maximum_input_size
            ):
                raise ValueError(
                    f"Tile {path} is {width}x{height}; direct tile inference supports "
                    f"at most {self.model_info.maximum_input_size}x"
                    f"{self.model_info.maximum_input_size}. "
                    "Use the upstream whole-slide runner for larger images."
                )
            array = np.asarray(image, dtype=np.uint8)
        patch_size = self.model_info.patch_size
        padded_height = ((height + patch_size - 1) // patch_size) * patch_size
        padded_width = ((width + patch_size - 1) // patch_size) * patch_size
        padded = np.full(
            (padded_height, padded_width, 3),
            255,
            dtype=np.uint8,
        )
        padded[:height, :width] = array
        tensor = torch.from_numpy(padded.copy()).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(self.model_info.normalization_mean).view(3, 1, 1)
        std = torch.tensor(self.model_info.normalization_std).view(3, 1, 1)
        tensor = ((tensor - mean) / std).unsqueeze(0)
        return tensor.to(device=self.device, dtype=self.dtype), width, height

    @staticmethod
    def _softmax_reorder(predictions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "nuclei_binary_map": F.softmax(
                predictions["nuclei_binary_map"], dim=1
            ).permute(0, 2, 3, 1).float().cpu(),
            "nuclei_type_map": F.softmax(
                predictions["nuclei_type_map"], dim=1
            ).permute(0, 2, 3, 1).float().cpu(),
            "hv_map": predictions["hv_map"].permute(0, 2, 3, 1).float().cpu(),
        }

    def _convert_cell(
        self, cell: dict[str, Any], width: int, height: int
    ) -> NucleusPrediction | None:
        type_id = int(cell["type"])
        probability = float(cell["type_prob"])
        if type_id not in PANNUKE_CLASS_NAMES or probability < self.min_type_probability:
            return None
        centroid_array = np.asarray(cell["centroid"], dtype=np.float64)
        if not (0 <= centroid_array[0] < width and 0 <= centroid_array[1] < height):
            return None
        contour = np.asarray(cell["contour"], dtype=np.float64)
        contour[:, 0] = np.clip(contour[:, 0], 0, width - 1)
        contour[:, 1] = np.clip(contour[:, 1], 0, height - 1)
        unique_contour = np.unique(contour, axis=0)
        if unique_contour.shape[0] < 3:
            return None
        bbox = np.asarray(cell["bbox"], dtype=np.float64)
        # Upstream bbox order is [[row_min, col_min], [row_max, col_max]].
        bbox_xy = (
            (
                float(np.clip(bbox[0, 1], 0, width - 1)),
                float(np.clip(bbox[0, 0], 0, height - 1)),
            ),
            (
                float(np.clip(bbox[1, 1], 0, width - 1)),
                float(np.clip(bbox[1, 0], 0, height - 1)),
            ),
        )
        return NucleusPrediction(
            contour=tuple((float(x), float(y)) for x, y in contour),
            type_id=type_id,
            type_probability=probability,
            centroid=(float(centroid_array[0]), float(centroid_array[1])),
            bbox=bbox_xy,
        )

    @torch.inference_mode()
    def annotate(self, image_path: str | Path) -> tuple[list[NucleusPrediction], int, int]:
        image_path = Path(image_path).expanduser().resolve()
        tensor, width, height = self._prepare_image(image_path)
        predictions = self.model(tensor, retrieve_tokens=False)
        postprocess_input = self._softmax_reorder(predictions)
        _, cell_dicts = self.postprocessor.post_process_batch(postprocess_input)
        converted = [
            prediction
            for cell in cell_dicts[0].values()
            if (prediction := self._convert_cell(cell, width, height)) is not None
        ]
        del tensor, predictions, postprocess_input
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()
        return converted, width, height


def _pair_tasks(pairs_path: Path, output_dir: Path) -> list[ImageTask]:
    records = [
        json.loads(line)
        for line in pairs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tasks: dict[tuple[Path, Path], ImageTask] = {}
    for record in records:
        stem = str(record["stem"])
        seed = int(record["seed"])
        seed_dir = Path(stem) / f"seed_{seed:010d}"
        intervention = record["intervention"]["slug"]
        pair_id = f"{stem}:seed={seed}:intervention={intervention}"
        pair_group_id = f"{stem}:seed={seed}"
        common = {
            "pair_group_id": pair_group_id,
            "source_tile_stem": stem,
            "generation_seed": seed,
            "pairs_manifest": str(pairs_path),
        }
        baseline_path = Path(record["baseline_image"]).expanduser().resolve()
        baseline_output = (output_dir / seed_dir / "baseline.geojson").resolve()
        baseline_key = (baseline_path, baseline_output)
        if baseline_key not in tasks:
            tasks[baseline_key] = ImageTask(
                baseline_path,
                baseline_output,
                {
                    **common,
                    "source_kind": "generated_baseline",
                    "pair_role": "baseline",
                    "intervention": None,
                },
            )
        counterfactual_path = Path(record["counterfactual_image"]).expanduser().resolve()
        counterfactual_output = (
            output_dir / seed_dir / f"{intervention}.geojson"
        ).resolve()
        tasks[(counterfactual_path, counterfactual_output)] = ImageTask(
            counterfactual_path,
            counterfactual_output,
            {
                **common,
                "source_kind": "generated_counterfactual",
                "pair_id": pair_id,
                "pair_role": "counterfactual",
                "intervention": record["intervention"],
            },
        )
    return list(tasks.values())


def _directory_tasks(input_dir: Path, output_dir: Path, recursive: bool) -> list[ImageTask]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    paths = sorted(
        path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    return [
        ImageTask(
            path,
            (output_dir / path.relative_to(input_dir).with_suffix(".geojson")).resolve(),
            {},
        )
        for path in paths
    ]


def _collect_tasks(args: argparse.Namespace, output_dir: Path) -> list[ImageTask]:
    if args.pairs_manifest is not None:
        pairs_path = args.pairs_manifest.expanduser().resolve()
        if not pairs_path.is_file():
            raise FileNotFoundError(f"Pairs manifest not found: {pairs_path}")
        tasks = _pair_tasks(pairs_path, output_dir)
    elif args.images:
        tasks = [
            ImageTask(
                path.expanduser().resolve(),
                (output_dir / f"{path.stem}.geojson").resolve(),
                {"source_kind": args.source_kind},
            )
            for path in args.images
        ]
    else:
        input_dir = (args.input_dir or DEFAULT_INPUT_DIR).expanduser().resolve()
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input image directory not found: {input_dir}")
        tasks = _directory_tasks(input_dir, output_dir, args.recursive)
    missing = [str(task.image_path) for task in tasks if not task.image_path.is_file()]
    if missing:
        raise FileNotFoundError(f"Input images are missing: {missing[:5]}")
    if not tasks:
        raise ValueError("No supported input images were found")
    output_paths = [task.output_path for task in tasks]
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("Two input images resolve to the same GeoJSON output path")
    if args.max_images is not None:
        tasks = tasks[: args.max_images]
    return tasks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input-dir", type=Path)
    inputs.add_argument("--image", action="append", type=Path, dest="images")
    inputs.add_argument("--pairs-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--source-kind", default="source_tile")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cellvit-root", type=Path, default=DEFAULT_CELLVIT_ROOT)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float16", "bfloat16", "float32"),
    )
    parser.add_argument("--magnification", type=int, default=40, choices=(20, 40))
    parser.add_argument("--mpp", type=float, default=0.25)
    parser.add_argument("--min-type-probability", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--sample", type=int, help="Randomly select this many input images.")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing output GeoJSON without loading CellViT++.",
    )
    args = parser.parse_args(argv)
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be at least one")
    if args.sample is not None and args.sample < 1:
        parser.error("--sample must be at least one")
    if not 0.0 <= args.min_type_probability <= 1.0:
        parser.error("--min-type-probability must be in [0, 1]")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    elif args.pairs_manifest is not None:
        output_dir = (
            args.pairs_manifest.expanduser().resolve().parent
            / "cellvit_plus_plus_annotations"
        )
    else:
        output_dir = DEFAULT_OUTPUT_DIR.resolve()
    tasks = _collect_tasks(args, output_dir)
    if args.sample is not None and args.sample < len(tasks):
        tasks = random.Random(args.sample_seed).sample(tasks, args.sample)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC)
    manifest_name = f"annotation_manifest_{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    manifest_path = output_dir / manifest_name
    model_path = args.model.expanduser().resolve()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "created_at": timestamp.isoformat(),
        "upstream": {
            "repository": "https://github.com/TIO-IKIM/CellViT-plus-plus",
            "commit": UPSTREAM_COMMIT,
            "source_root": str(args.cellvit_root.expanduser().resolve()),
        },
        "model": {
            "path": str(model_path),
            "sha256": _sha256(model_path) if model_path.is_file() else None,
        },
        "configuration": {
            "requested_device": args.device,
            "requested_dtype": args.dtype,
            "magnification": args.magnification,
            "mpp": args.mpp,
            "min_type_probability": args.min_type_probability,
            "allow_empty": args.allow_empty,
        },
        "inputs": [],
    }
    _json_write(manifest_path, manifest)

    annotator: CellViTTileAnnotator | None = None
    results: list[dict[str, Any]] = []
    try:
        for task in tasks:
            started = time.monotonic()
            with Image.open(task.image_path) as image:
                width, height = image.size
            if task.output_path.exists() and not args.overwrite:
                _, summary = load_and_validate_geojson(
                    task.output_path,
                    image_width=width,
                    image_height=height,
                    allow_empty=args.allow_empty,
                    strict_bounds=False,
                )
                status = "validated_existing"
            elif args.validate_only:
                raise FileNotFoundError(
                    f"No existing output to validate: {task.output_path}"
                )
            else:
                if annotator is None:
                    annotator = CellViTTileAnnotator(
                        model_path,
                        args.cellvit_root,
                        device=args.device,
                        dtype=args.dtype,
                        magnification=args.magnification,
                        min_type_probability=args.min_type_probability,
                    )
                    manifest["configuration"].update(
                        {
                            "resolved_device": str(annotator.device),
                            "resolved_dtype": str(annotator.dtype),
                            "model_info": annotator.model_info.__dict__,
                        }
                    )
                    _json_write(manifest_path, manifest)
                predictions, predicted_width, predicted_height = annotator.annotate(
                    task.image_path
                )
                if (predicted_width, predicted_height) != (width, height):
                    raise RuntimeError("CellViT++ image dimensions changed unexpectedly")
                payload = predictions_to_geojson(
                    predictions,
                    image_path=task.image_path,
                    image_width=width,
                    image_height=height,
                    model={
                        "name": "CellViT-SAM-H-x40-AMP-001",
                        "checkpoint_sha256": manifest["model"]["sha256"],
                        "upstream_commit": UPSTREAM_COMMIT,
                        "magnification": args.magnification,
                        "mpp": args.mpp,
                    },
                    source_metadata={
                        **task.source_metadata,
                        "source_kind": task.source_metadata.get(
                            "source_kind", args.source_kind
                        ),
                    },
                )
                summary = validate_geojson(
                    payload,
                    image_width=width,
                    image_height=height,
                    allow_empty=args.allow_empty,
                )
                write_geojson(task.output_path, payload)
                status = "annotated"
            result = {
                "status": status,
                "image_path": str(task.image_path),
                "image_sha256": _sha256(task.image_path),
                "output_geojson": str(task.output_path),
                "image_width": width,
                "image_height": height,
                "source_metadata": task.source_metadata,
                "summary": summary,
                "elapsed_seconds": time.monotonic() - started,
            }
            results.append(result)
            print(
                f"{status}: {task.image_path.name} -> {summary['nucleus_count']} nuclei"
            )
            manifest["inputs"] = results
            _json_write(manifest_path, manifest)
    except Exception:
        manifest["status"] = "failed"
        manifest["inputs"] = results
        manifest["failed_at"] = datetime.now(UTC).isoformat()
        _json_write(manifest_path, manifest)
        raise

    aggregate_counts: Counter[str] = Counter()
    for result in results:
        aggregate_counts.update(result["summary"]["class_counts"])
    manifest["status"] = "completed"
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    manifest["inputs"] = results
    manifest["summary"] = {
        "image_count": len(results),
        "nucleus_count": sum(
            result["summary"]["nucleus_count"] for result in results
        ),
        "class_counts": dict(sorted(aggregate_counts.items())),
    }
    _json_write(manifest_path, manifest)
    print(f"Annotation complete: {manifest_path}")


if __name__ == "__main__":
    main()
