"""Small YAML-to-argparse bridge used by the training workflow entry points."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def parse_args_with_yaml(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Apply a flat YAML mapping as parser defaults, then parse CLI overrides."""
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path)
    known, _ = preliminary.parse_known_args(argv)
    if known.config is not None:
        try:
            import yaml
        except ImportError as error:
            raise ImportError("Reading --config requires PyYAML") from error
        config_path = known.config.expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Training config not found: {config_path}")
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Training config must be a YAML mapping: {config_path}")
        valid = {action.dest for action in parser._actions}
        unknown = sorted(set(payload).difference(valid))
        if unknown:
            raise ValueError(f"Unknown training config keys: {unknown}")
        parser.set_defaults(**payload)
    return parser.parse_args(argv)


def resolve_repository_path(value: str | Path, repository_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repository_root / path
    return path.resolve()


def resolve_model_source(value: str, repository_root: Path) -> str:
    """Resolve repository-relative model paths while preserving Hub identifiers."""
    candidate = (repository_root / value).expanduser()
    if not Path(value).is_absolute() and candidate.exists():
        return str(candidate.resolve())
    absolute = Path(value).expanduser()
    if absolute.is_absolute():
        return str(absolute.resolve())
    return value
