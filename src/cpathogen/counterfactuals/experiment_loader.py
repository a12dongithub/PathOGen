"""Dynamic loading and validation of Python-only experiment modules."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType

from .interventions import ConditionIntervention


def _load_module(module_or_path: str) -> ModuleType:
    path = Path(module_or_path).expanduser()
    if path.suffix == ".py" or path.is_file():
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Experiment module not found: {path}")
        spec = importlib.util.spec_from_file_location(
            f"cpathogen_experiment_{path.stem}", path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load experiment module: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(module_or_path)


def load_interventions(module_or_path: str) -> tuple[ModuleType, list[ConditionIntervention]]:
    """Load ``build_interventions()`` and enforce the plugin contract."""
    module = _load_module(module_or_path)
    builder = getattr(module, "build_interventions", None)
    if not callable(builder):
        raise TypeError(
            f"Experiment {module.__name__} must define build_interventions()"
        )
    interventions = list(builder())
    if not interventions:
        raise ValueError(f"Experiment {module.__name__} returned no interventions")
    invalid = [item for item in interventions if not isinstance(item, ConditionIntervention)]
    if invalid:
        raise TypeError("Every experiment item must inherit ConditionIntervention")
    slugs = [item.slug for item in interventions]
    duplicates = sorted({slug for slug in slugs if slugs.count(slug) > 1})
    if duplicates:
        raise ValueError(f"Experiment intervention slugs must be unique: {duplicates}")
    return module, interventions


def select_interventions(
    interventions: list[ConditionIntervention], selected: list[str] | None
) -> list[ConditionIntervention]:
    if not selected:
        return interventions
    requested = {value.lower() for value in selected}
    available = {item.slug: item for item in interventions}
    missing = sorted(requested.difference(available))
    if missing:
        raise ValueError(
            f"Unknown intervention(s) {missing}; available: {sorted(available)}"
        )
    return [item for item in interventions if item.slug in requested]
