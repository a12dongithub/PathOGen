"""CellViT++ inference adapter used by fidelity experiments."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .constants import CELL_COLORS, CELL_TYPES


class CellViTRunner:
    def __init__(
        self,
        project_root: Path,
        model_path: Path,
        gpu: int = 0,
        precision: str = "auto",
    ):
        self.project_root = project_root.resolve()
        self.model_path = model_path.resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"CellViT checkpoint missing: {self.model_path}")
        self.device = torch.device(
            f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        )
        self.gpu_name = torch.cuda.get_device_name(gpu) if self.device.type == "cuda" else ""
        if precision not in {"auto", "fp16", "fp32"}:
            raise ValueError("CellViT precision must be auto, fp16, or fp32")
        if precision == "auto":
            precision = (
                "fp32"
                if self.device.type == "cpu" or "GTX 16" in self.gpu_name.upper()
                else "fp16"
            )
        self.precision = precision
        self.model = None
        self.postprocessor = None
        self.architecture = ""

    def _load(self) -> None:
        if self.model is not None:
            return
        sys.path.insert(0, str(self.project_root))
        from cellvit.models.cell_segmentation.cellvit import CellViT
        from cellvit.models.cell_segmentation.cellvit_256 import CellViT256
        from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
        from cellvit.models.cell_segmentation.cellvit_uni import CellViTUNI
        from cellvit.models.cell_segmentation.postprocessing import (
            DetectionCellPostProcessor,
        )
        from cellvit.utils.tools import unflatten_dict

        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        run_conf = unflatten_dict(checkpoint["config"], ".")
        architecture = checkpoint["arch"]
        if architecture == "CellViT":
            model = CellViT(
                num_nuclei_classes=run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=run_conf["data"]["num_tissue_classes"],
                embed_dim=run_conf["model"]["embed_dim"],
                input_channels=run_conf["model"].get("input_channels", 3),
                depth=run_conf["model"]["depth"],
                num_heads=run_conf["model"]["num_heads"],
                extract_layers=run_conf["model"]["extract_layers"],
                regression_loss=run_conf["model"].get("regression_loss", False),
            )
        elif architecture == "CellViT256":
            model = CellViT256(
                model256_path=None,
                num_nuclei_classes=run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=run_conf["data"]["num_tissue_classes"],
                regression_loss=run_conf["model"].get("regression_loss", False),
            )
        elif architecture == "CellViTSAM":
            model = CellViTSAM(
                model_path=None,
                num_nuclei_classes=run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=run_conf["data"]["num_tissue_classes"],
                vit_structure=run_conf["model"]["backbone"],
                regression_loss=run_conf["model"].get("regression_loss", False),
            )
        elif architecture == "CellViTUNI":
            model = CellViTUNI(
                model_uni_path=None,
                num_nuclei_classes=run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=run_conf["data"]["num_tissue_classes"],
            )
        else:
            raise NotImplementedError(f"Unsupported CellViT architecture: {architecture}")
        model.load_state_dict(checkpoint["model_state_dict"])
        self.model = model.eval().to(self.device)
        self.postprocessor = DetectionCellPostProcessor(
            nr_types=run_conf["data"]["num_nuclei_classes"], magnification=40
        )
        self.architecture = architecture

    def describe(self) -> dict[str, str]:
        return {
            "project_root": str(self.project_root),
            "model": str(self.model_path),
            "architecture": self.architecture,
            "device": str(self.device),
            "gpu": self.gpu_name,
            "precision": self.precision,
        }

    def _tensor(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
        return ((tensor - 0.5) / 0.5).to(self.device)

    @staticmethod
    def _prepare(predictions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        predictions["nuclei_binary_map"] = F.softmax(
            predictions["nuclei_binary_map"], dim=1
        ).permute(0, 2, 3, 1).float()
        predictions["nuclei_type_map"] = F.softmax(
            predictions["nuclei_type_map"], dim=1
        ).permute(0, 2, 3, 1).float()
        predictions["hv_map"] = predictions["hv_map"].permute(0, 2, 3, 1).float()
        return predictions

    def infer_batch(self, images: list[Image.Image]) -> list[dict]:
        """Run true batched segmentation, reducing the batch automatically on OOM."""
        if not images:
            return []
        self._load()
        assert self.model is not None and self.postprocessor is not None
        tensor = torch.cat([self._tensor(image) for image in images], dim=0)
        use_amp = self.device.type == "cuda" and self.precision == "fp16"
        split_at = None
        try:
            with torch.inference_mode():
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        predictions = self.model.forward(tensor, retrieve_tokens=True)
                else:
                    predictions = self.model.forward(tensor, retrieve_tokens=True)
        except RuntimeError as error:
            is_oom = isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(
                error
            ).lower()
            if not is_oom or len(images) == 1:
                raise
            split_at = len(images) // 2
        if split_at is not None:
            # Leave the exception block before retrying so failed-forward
            # tensors retained by its traceback can be reclaimed.
            del tensor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"[cellvit] CUDA OOM at batch {len(images)}; retrying "
                f"as {split_at}+{len(images) - split_at}",
                flush=True,
            )
            return self.infer_batch(images[:split_at]) + self.infer_batch(
                images[split_at:]
            )
        required = ("nuclei_binary_map", "nuclei_type_map", "hv_map")
        if not all(torch.isfinite(predictions[name]).all() for name in required):
            if not use_amp:
                raise RuntimeError("CellViT produced non-finite predictions in FP32")
            self.precision = "fp32"
            with torch.inference_mode():
                predictions = self.model.forward(tensor, retrieve_tokens=True)
            if not all(torch.isfinite(predictions[name]).all() for name in required):
                raise RuntimeError("CellViT produced non-finite predictions after FP32 retry")
        predictions = self._prepare(predictions)
        _, cell_dicts = self.postprocessor.post_process_batch(predictions)
        return cell_dicts

    def infer(self, image: Image.Image) -> dict:
        return self.infer_batch([image])[0]

    def unload(self) -> None:
        self.model = None
        self.postprocessor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def save_cellvit_geojson(cells: dict, destination: Path) -> None:
    features = []
    for cell_id, cell in cells.items():
        raw_type = int(cell["type"])
        channel = raw_type - 1
        if channel not in CELL_TYPES:
            continue
        cell_name = CELL_TYPES[channel]
        contour = np.asarray(cell["contour"], dtype=float).round(3).tolist()
        if contour and contour[0] != contour[-1]:
            contour.append(contour[0])
        features.append(
            {
                "type": "Feature",
                "id": str(cell_id),
                "geometry": {"type": "Polygon", "coordinates": [contour]},
                "properties": {
                    "classification": {
                        "id": raw_type,
                        "name": cell_name,
                        "color": list(CELL_COLORS[cell_name]),
                    },
                    "centroid": np.asarray(cell["centroid"], dtype=float).round(3).tolist(),
                    "type_probability": round(float(cell["type_prob"]), 6),
                },
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )
