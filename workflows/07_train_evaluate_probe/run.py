#!/usr/bin/env python3
"""Train a frozen CTransPath linear probe and score counterfactual tiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import timm_ctp
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

TASK_NAME = "BCSS/TIGER invasive tumor vs tumor-associated stroma"
POSITIVE_CLASS = "invasive_tumor"
NEGATIVE_CLASS = "tumor_associated_stroma"
LABEL_TO_INT = {NEGATIVE_CLASS: 0, POSITIVE_CLASS: 1}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--counterfactual-root", type=Path, required=True)
    parser.add_argument(
        "--counterfactual-source-uri",
        help="GCS prefix recorded in the final prediction table.",
    )
    parser.add_argument("--expected-counterfactual-candidates", type=int)
    parser.add_argument("--counterfactual-archive-member-prefix", default="")
    parser.add_argument("--ctranspath-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cuda", "cpu", "mps")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=(0.01, 0.1, 1.0, 10.0),
        help="L2 logistic-regression strengths selected on validation AUROC.",
    )
    parser.add_argument(
        "--smoke-limit-per-class-per-split",
        type=int,
        help="Developer-only balanced row limit for a fast local smoke test.",
    )
    parser.add_argument(
        "--smoke-limit-counterfactual",
        type=int,
        help="Developer-only counterfactual row limit.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ConvStem(nn.Module):
    """Convolutional patch stem required by the released CTransPath weights."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Any | None = None,
        flatten: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        image_size = (
            (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        )
        patch_dimensions = (
            (patch_size, patch_size)
            if isinstance(patch_size, int)
            else tuple(patch_size)
        )
        if patch_dimensions != (4, 4) or in_chans != 3:
            raise ValueError("CTransPath expects RGB input and patch_size=4")
        self.img_size = image_size
        self.patch_size = patch_dimensions
        self.grid_size = (
            image_size[0] // patch_dimensions[0],
            image_size[1] // patch_dimensions[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        stem: list[nn.Module] = []
        input_dim, output_dim = in_chans, embed_dim // 8
        for _ in range(2):
            stem.extend(
                (
                    nn.Conv2d(
                        input_dim,
                        output_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(output_dim),
                    nn.ReLU(inplace=True),
                )
            )
            input_dim, output_dim = output_dim, output_dim * 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.proj(inputs)
        if self.flatten:
            features = features.flatten(2).transpose(1, 2)
        return self.norm(features)


def build_encoder(checkpoint: Path, device: torch.device) -> nn.Module:
    model = timm_ctp.create_model(
        "swin_tiny_patch4_window7_224",
        embed_layer=ConvStem,
        pretrained=False,
    )
    model.head = nn.Identity()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = (
        payload["model"]
        if isinstance(payload, dict) and "model" in payload
        else payload
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.requires_grad_(False)
    return model.to(device)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def image_transform() -> transforms.Compose:
    return transforms.Compose(
        (
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        )
    )


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.transform = image_transform()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def extract_embeddings(
    model: nn.Module,
    paths: list[Path],
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    description: str,
) -> np.ndarray:
    loader = DataLoader(
        ImageDataset(paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for images in tqdm(loader, desc=description):
            embeddings = model(images.to(device, non_blocking=True))
            embeddings = torch.nn.functional.normalize(embeddings, dim=1)
            chunks.append(embeddings.cpu().numpy().astype(np.float32, copy=False))
    if not chunks:
        raise ValueError(f"No images supplied for {description}")
    return np.concatenate(chunks, axis=0)


def resolve_training_paths(frame: pd.DataFrame, root: Path) -> list[Path]:
    paths = [root / "images" / Path(value).name for value in frame["image_path"]]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} training image(s), e.g. {missing[0]}"
        )
    return paths


def resolve_counterfactual_path(value: str, root: Path) -> Path:
    path = Path(value)
    parts = path.parts
    if "images" not in parts:
        raise ValueError(f"Counterfactual path has no images/ component: {value}")
    relative = Path(*parts[parts.index("images") :])
    return root / relative


def patient_from_stem(stem: str) -> str | None:
    parts = stem.split("_")[0].split("-")
    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])
    return None


def knob_sd(parameters: str, condition: str) -> float:
    if condition == "baseline":
        return 0.0
    try:
        value = json.loads(parameters).get("sd_steps")
        if value is not None:
            return float(value)
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
        pass
    raise ValueError(f"No sd_steps found for condition {condition!r}")


def binary_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(np.int64)
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "f1": float(f1_score(y_true, prediction)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
    }


def fit_head(
    embeddings: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    c_values: list[float],
    seed: int,
) -> tuple[LogisticRegression, dict[str, Any]]:
    train = splits == "train"
    validation = splits == "validation"
    test = splits == "test"
    candidates: list[dict[str, Any]] = []
    for c_value in c_values:
        classifier = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=2_000,
            random_state=seed,
        )
        classifier.fit(embeddings[train], labels[train])
        probability = classifier.predict_proba(embeddings[validation])[:, 1]
        candidates.append(
            {"C": c_value, **binary_metrics(labels[validation], probability)}
        )
    selected = max(candidates, key=lambda row: (row["roc_auc"], -row["log_loss"]))
    development = train | validation
    final = LogisticRegression(
        C=float(selected["C"]),
        class_weight="balanced",
        max_iter=2_000,
        random_state=seed,
    )
    final.fit(embeddings[development], labels[development])
    result = {
        "selection": candidates,
        "selected_C": float(selected["C"]),
        "test": binary_metrics(
            labels[test], final.predict_proba(embeddings[test])[:, 1]
        ),
        "counts": {
            split: int(np.sum(splits == split))
            for split in ("train", "validation", "test")
        },
    }
    return final, result


def balanced_smoke_subset(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    rows = []
    for (_, _), group in frame.groupby(["split", "label"], sort=False):
        rows.append(group.sample(n=min(limit, len(group)), random_state=seed))
    return pd.concat(rows, ignore_index=True)


@dataclass(frozen=True)
class Inputs:
    training_manifest: Path
    counterfactual_manifest: Path


def validate_inputs(args: argparse.Namespace) -> Inputs:
    training_manifest = args.training_root / "tiles.csv"
    counterfactual_manifest = args.counterfactual_root / "images.csv"
    for path in (
        training_manifest,
        counterfactual_manifest,
        args.ctranspath_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {args.output_dir}")
    return Inputs(training_manifest, counterfactual_manifest)


def main() -> None:
    args = parse_args()
    inputs = validate_inputs(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    counterfactuals = pd.read_csv(inputs.counterfactual_manifest)
    if args.smoke_limit_counterfactual:
        counterfactuals = counterfactuals.head(args.smoke_limit_counterfactual).copy()
    counterfactuals["source_patient_id"] = counterfactuals["stem"].map(
        patient_from_stem
    )
    counterfactual_patients = set(counterfactuals["source_patient_id"].dropna())

    training = pd.read_csv(inputs.training_manifest)
    training = training[training["split"].isin(("train", "validation", "test"))].copy()
    training_rows_before_exclusion = len(training)
    training_patients_before_exclusion = int(training["patient_id"].nunique())
    excluded_patients = sorted(set(training["patient_id"]) & counterfactual_patients)
    training = training.loc[~training["patient_id"].isin(excluded_patients)].copy()
    if set(training["split"]) != {"train", "validation", "test"}:
        raise ValueError("Counterfactual-patient exclusion emptied a real-data split")
    for split, group in training.groupby("split"):
        if set(group["label"]) != set(LABEL_TO_INT):
            raise ValueError(f"Real-data split {split!r} lost one class after exclusion")
    if set(training["label"]) != set(LABEL_TO_INT):
        raise ValueError(
            f"Unexpected training labels: {sorted(training['label'].unique())}"
        )
    if args.smoke_limit_per_class_per_split:
        training = balanced_smoke_subset(
            training, args.smoke_limit_per_class_per_split, args.seed
        )
    training_paths = resolve_training_paths(training, args.training_root)
    labels = training["label"].map(LABEL_TO_INT).to_numpy(dtype=np.int64)
    splits = training["split"].to_numpy()

    encoder = build_encoder(args.ctranspath_checkpoint, device)
    train_embeddings = extract_embeddings(
        encoder,
        training_paths,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        description="real BCSS tiles",
    )
    np.savez_compressed(
        args.output_dir / "real_tile_embeddings.npz",
        embeddings=train_embeddings,
        tile_id=training["tile_id"].to_numpy(),
        label=labels,
        split=splits,
    )
    classifier, metrics = fit_head(
        train_embeddings, labels, splits, list(args.c_values), args.seed
    )
    joblib.dump(
        {
            "classifier": classifier,
            "positive_class": POSITIVE_CLASS,
            "negative_class": NEGATIVE_CLASS,
            "task": TASK_NAME,
        },
        args.output_dir / "classifier.joblib",
    )
    np.savez(
        args.output_dir / "head_weights.npz",
        coef=classifier.coef_,
        intercept=classifier.intercept_,
        classes=classifier.classes_,
    )

    counterfactual_paths = [
        resolve_counterfactual_path(value, args.counterfactual_root)
        for value in counterfactuals["image_path"]
    ]
    missing = [path for path in counterfactual_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} counterfactual image(s), e.g. {missing[0]}"
        )
    cf_embeddings = extract_embeddings(
        encoder,
        counterfactual_paths,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        description="counterfactual tiles",
    )
    probabilities = classifier.predict_proba(cf_embeddings)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int64)
    results = counterfactuals.copy()
    results["model_id"] = "ctranspath_frozen_l2_logistic"
    results["task"] = TASK_NAME
    results["source_patient_id"] = counterfactuals["source_patient_id"]
    results["knob_sd"] = [
        knob_sd(parameters, condition)
        for parameters, condition in zip(
            results["intervention_parameters"], results["condition"], strict=True
        )
    ]
    results["probability_invasive_tumor"] = probabilities
    results["probability_tumor_associated_stroma"] = 1.0 - probabilities
    results["predicted_class_index"] = predictions
    results["predicted_label"] = np.where(
        predictions == 1, POSITIVE_CLASS, NEGATIVE_CLASS
    )
    results["local_image_path"] = [str(path) for path in counterfactual_paths]
    results["relative_image_path"] = [
        str(path.relative_to(args.counterfactual_root)) for path in counterfactual_paths
    ]
    if args.counterfactual_source_uri:
        source_prefix = args.counterfactual_source_uri.rstrip("/")
        results["counterfactual_source_uri"] = source_prefix
        if source_prefix.endswith(".zip"):
            archive_prefix = Path(args.counterfactual_archive_member_prefix)
            results["counterfactual_archive_member"] = [
                str(archive_prefix / relative)
                for relative in results["relative_image_path"]
            ]
            results["counterfactual_gcs_uri"] = pd.NA
        else:
            results["counterfactual_archive_member"] = pd.NA
            results["counterfactual_gcs_uri"] = [
                f"{source_prefix}/{relative}"
                for relative in results["relative_image_path"]
            ]
    else:
        results["counterfactual_source_uri"] = pd.NA
        results["counterfactual_archive_member"] = pd.NA
        results["counterfactual_gcs_uri"] = pd.NA

    if not args.smoke_limit_counterfactual:
        counts = results.groupby("candidate_id")["condition"].nunique()
        candidate_count = int(results["candidate_id"].nunique())
        if (
            args.expected_counterfactual_candidates
            and candidate_count != args.expected_counterfactual_candidates
        ):
            raise ValueError(
                f"Expected {args.expected_counterfactual_candidates} candidates; "
                f"found {candidate_count}"
            )
        if len(results) != 4 * candidate_count:
            raise ValueError(
                f"Expected four rows per candidate; found {len(results)} rows for "
                f"{candidate_count} candidates"
            )
        if not (counts == 4).all() or set(results["knob_sd"]) != {0.0, 0.5, 1.0, 1.5}:
            raise ValueError(
                "Every candidate must have exactly the four declared SD levels"
            )

    trained_patients = set(training["patient_id"])
    evaluated_patients = set(results["source_patient_id"].dropna())
    overlap = sorted(trained_patients & evaluated_patients)
    if overlap:
        raise ValueError(
            "Counterfactual source patients leak into head development/test splits: "
            + ", ".join(overlap[:5])
        )
    baseline = (
        results.loc[
            results["condition"] == "baseline",
            ["candidate_id", "probability_invasive_tumor"],
        ]
        .drop_duplicates("candidate_id")
        .rename(
            columns={
                "probability_invasive_tumor": "baseline_probability_invasive_tumor"
            }
        )
    )
    results = results.merge(
        baseline, on="candidate_id", how="left", validate="many_to_one"
    )
    results["delta_probability_invasive_tumor"] = (
        results["probability_invasive_tumor"]
        - results["baseline_probability_invasive_tumor"]
    )
    results.to_csv(args.output_dir / "counterfactual_predictions.csv", index=False)
    results.to_parquet(
        args.output_dir / "counterfactual_predictions.parquet", index=False
    )
    summary = (
        results.groupby(["condition", "knob_sd"], as_index=False)
        .agg(
            tile_count=("candidate_id", "size"),
            mean_probability_invasive_tumor=("probability_invasive_tumor", "mean"),
            std_probability_invasive_tumor=("probability_invasive_tumor", "std"),
            mean_delta_probability_invasive_tumor=(
                "delta_probability_invasive_tumor",
                "mean",
            ),
        )
        .sort_values("knob_sd")
    )
    summary.to_csv(args.output_dir / "counterfactual_summary.csv", index=False)

    completed_at = datetime.now(timezone.utc).isoformat()
    metrics.update(
        {
            "task": TASK_NAME,
            "positive_class": POSITIVE_CLASS,
            "encoder": "CTransPath",
            "encoder_checkpoint_sha256": sha256(args.ctranspath_checkpoint),
            "device": str(device),
            "counterfactual_rows": len(results),
            "counterfactual_candidates": int(results["candidate_id"].nunique()),
            "counterfactual_source_patients": len(counterfactual_patients),
            "training_exclusion": {
                "reason": "counterfactual source-patient disjointness",
                "excluded_patient_count": len(excluded_patients),
                "excluded_patients": excluded_patients,
                "rows_before": training_rows_before_exclusion,
                "rows_after": len(training),
                "patients_before": training_patients_before_exclusion,
                "patients_after": int(training["patient_id"].nunique()),
            },
            "completed_at": completed_at,
        }
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "created_at": completed_at,
        "task": TASK_NAME,
        "encoder": "CTransPath",
        "encoder_frozen": True,
        "head": "L2-regularized logistic regression",
        "counterfactual_source_uri": args.counterfactual_source_uri,
        "counterfactual_archive_member_prefix": (
            args.counterfactual_archive_member_prefix or None
        ),
        "training_manifest_sha256": sha256(inputs.training_manifest),
        "counterfactual_manifest_sha256": sha256(inputs.counterfactual_manifest),
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(args.output_dir.iterdir())
            if path.is_file() and path.name != "run_manifest.json"
        },
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {len(results)} counterfactual predictions to {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
