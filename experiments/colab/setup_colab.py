#!/usr/bin/env python
"""Install dependencies and prepare every PathOGen fidelity asset in Colab."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.colab.layout import (
    DEFAULT_ASSET_ROOT,
    REPO_ROOT,
    RuntimePaths,
    find_cellvit_model,
    find_cellvit_source,
    find_checkpoint,
    find_dataset,
)

DATA_URL = "https://drive.google.com/file/d/1sBc4-CexT3S2cw1LZysrX4BLjVjN6BPt/view?usp=sharing"
MODEL_URL = "https://drive.google.com/file/d/1QLymjt0qnjM2FM-oR5vRYB0B1URcg5wq/view?usp=sharing"
CELLVIT_GIT_URL = "https://github.com/TIO-IKIM/CellViT-plus-plus.git"
CELLVIT_GIT_REF = "463c5c44bfdebfbe3943597eaa84daf3f5e26a5f"
MIN_DATA_PREP_FREE_GIB = 48.0
MIN_MODEL_PREP_FREE_GIB = 12.0
MIN_BOTH_PREP_FREE_GIB = 55.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--cellvit-root", type=Path)
    parser.add_argument("--cellvit-model", type=Path)
    parser.add_argument("--cellvit-model-url")
    parser.add_argument("--cellvit-model-name", default="CellViT-256-x40-AMP.pth")
    parser.add_argument("--data-url", default=DATA_URL)
    parser.add_argument("--model-url", default=MODEL_URL)
    parser.add_argument("--cellvit-git-url", default=CELLVIT_GIT_URL)
    parser.add_argument("--cellvit-git-ref", default=CELLVIT_GIT_REF)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--require-cellvit-model", action="store_true")
    return parser.parse_args()


def gib(value: int) -> float:
    return value / (1024**3)


def free_gib(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return gib(shutil.disk_usage(path).free)


def install_dependencies() -> None:
    requirements = REPO_ROOT / "experiments" / "requirements_fidelity.txt"
    print(f"[setup] Installing {requirements}", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)],
        check=True,
    )


def download_drive_file(url: str, destination: Path) -> Path:
    import gdown

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"[assets] Resuming/reusing {destination}", flush=True)
    result = gdown.download(
        url=url,
        output=str(destination),
        quiet=False,
        fuzzy=True,
        resume=True,
    )
    if not result or not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(
            f"Google Drive download failed for {url}. Confirm link sharing permits downloads."
        )
    print(f"[assets] Downloaded {destination.name}: {gib(destination.stat().st_size):.2f} GiB")
    return destination


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    print(f"[assets] Extracting {archive} -> {destination}", flush=True)
    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()
        for member in members:
            target = (destination / member.filename).resolve()
            if target != destination_root and destination_root not in target.parents:
                raise RuntimeError(f"Unsafe ZIP member path: {member.filename}")
        for index, member in enumerate(members, start=1):
            handle.extract(member, destination)
            if index % 10_000 == 0 or index == len(members):
                print(f"[assets] Extracted {index:,}/{len(members):,} entries", flush=True)


def locate_or_prepare_zip(
    search_root: Path,
    archive: Path,
    url: str,
    finder: Callable[[Path], Path],
    keep_archive: bool,
) -> Path:
    try:
        resolved = finder(search_root)
        print(f"[assets] Reusing existing asset: {resolved}")
        return resolved
    except FileNotFoundError:
        pass
    downloaded = download_drive_file(url, archive)
    safe_extract_zip(downloaded, search_root)
    resolved = finder(search_root)
    if not keep_archive:
        downloaded.unlink(missing_ok=True)
        print(f"[assets] Deleted extracted archive: {downloaded}")
    return resolved


def prepare_cellvit_source(root: Path, git_url: str, git_ref: str) -> Path:
    try:
        source = find_cellvit_source(root)
        print(f"[cellvit] Reusing source: {source}")
        return source
    except FileNotFoundError:
        pass
    destination = root / "repository"
    if destination.exists():
        raise RuntimeError(
            f"Incomplete CellViT++ checkout at {destination}. Remove that child directory "
            "or pass --cellvit-root pointing to a valid checkout."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[cellvit] Sparse-cloning {git_url} -> {destination}", flush=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            git_url,
            str(destination),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "fetch", "--depth", "1", "origin", git_ref],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "sparse-checkout", "set", "cellvit"],
        check=True,
    )
    return find_cellvit_source(destination)


def git_head(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def resolve_cellvit_model(args: argparse.Namespace, asset_root: Path) -> Path | None:
    if args.cellvit_model is not None:
        return find_cellvit_model(args.cellvit_model, args.cellvit_model_name)
    model_root = asset_root / "checkpoints" / "cellvit"
    if args.cellvit_model_url:
        destination = model_root / args.cellvit_model_name
        download_drive_file(args.cellvit_model_url, destination)
    try:
        return find_cellvit_model(model_root, args.cellvit_model_name)
    except FileNotFoundError:
        if args.require_cellvit_model:
            raise
        print(
            "[cellvit] WARNING: no CellViT++ checkpoint found. Upload one to "
            f"{model_root} or rerun with --cellvit-model/--cellvit-model-url."
        )
        return None


def report_runtime() -> None:
    import torch

    print(f"[runtime] Python: {sys.version.split()[0]}")
    print(f"[runtime] PyTorch: {torch.__version__}")
    print(f"[runtime] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        print(
            f"[runtime] GPU: {properties.name} ({properties.total_memory / (1024**3):.1f} GiB)"
        )
    else:
        print("[runtime] WARNING: generation requires a Colab GPU runtime")


def main() -> None:
    args = parse_args()
    if not args.skip_install:
        install_dependencies()
    report_runtime()

    asset_root = args.asset_root.expanduser().resolve()
    data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root
        else asset_root / "data"
    )
    checkpoint_root = (
        args.checkpoint_root.expanduser().resolve()
        if args.checkpoint_root
        else asset_root / "checkpoints" / "pathogen"
    )
    cellvit_search_root = (
        args.cellvit_root.expanduser().resolve()
        if args.cellvit_root
        else asset_root / "external" / "CellViT-plus-plus"
    )

    data_missing = False
    checkpoint_missing = False
    try:
        find_dataset(data_root)
    except FileNotFoundError:
        data_missing = True
    try:
        find_checkpoint(checkpoint_root)
    except FileNotFoundError:
        checkpoint_missing = True
    available = free_gib(asset_root)
    required = 0.0
    if data_missing and checkpoint_missing:
        required = MIN_BOTH_PREP_FREE_GIB
    elif data_missing:
        required = MIN_DATA_PREP_FREE_GIB
    elif checkpoint_missing:
        required = MIN_MODEL_PREP_FREE_GIB
    print(f"[disk] Free space: {available:.2f} GiB; required for missing assets: {required:.0f} GiB")
    if available < required:
        raise RuntimeError(
            f"Insufficient disk space: need {required:.0f} GiB, have {available:.2f} GiB"
        )

    downloads = asset_root / "downloads"
    data_dir = locate_or_prepare_zip(
        data_root,
        downloads / "512_final_dataset.zip",
        args.data_url,
        find_dataset,
        args.keep_archives,
    )
    checkpoint_dir = locate_or_prepare_zip(
        checkpoint_root,
        downloads / "checkpoint-30000_FID58.zip",
        args.model_url,
        find_checkpoint,
        args.keep_archives,
    )
    cellvit_root = prepare_cellvit_source(
        cellvit_search_root, args.cellvit_git_url, args.cellvit_git_ref
    )
    cellvit_model = resolve_cellvit_model(args, asset_root)
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else asset_root / "outputs"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    paths = RuntimePaths(
        repo_root=REPO_ROOT.resolve(),
        asset_root=asset_root,
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        cellvit_root=cellvit_root,
        cellvit_commit=git_head(cellvit_root),
        cellvit_model=cellvit_model,
        output_root=output_root,
    )
    config = args.config.expanduser().resolve() if args.config else asset_root / "runtime_paths.json"
    paths.write(config)
    print(f"[setup] Runtime configuration written to {config}")
    for key, value in paths.to_payload().items():
        print(f"[setup] {key}: {value}")
    print("[setup] Base Stable Diffusion files download from Hugging Face on first generation.")


if __name__ == "__main__":
    main()
