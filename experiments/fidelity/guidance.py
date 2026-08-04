"""Extension points for classifier- or CellViT-guided PathOGen sampling."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class GenerationContext:
    stem: str
    condition_id: str
    spatial_map: np.ndarray
    morphology: np.ndarray
    seed: int
    attempt: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self, **updates: Any) -> GenerationContext:
        return replace(self, **updates)


@dataclass(frozen=True)
class CandidateDecision:
    accept: bool = True
    score: float | None = None
    reason: str = "accepted"
    next_morphology_delta: tuple[float, ...] | None = None
    next_spatial_scale: float | None = None


class GuidanceHook:
    """Override any method to add control adaptation or sampling guidance.

    `adjust_conditions` can change morphology/spatial controls before sampling.
    `on_denoising_step` can alter latents for gradient-based guidance.
    `evaluate_candidate` can score/reject decoded samples and request another attempt.
    """

    def adjust_conditions(self, context: GenerationContext) -> GenerationContext:
        return context

    def on_denoising_step(
        self,
        context: GenerationContext,
        step_index: int,
        timestep: Any,
        latents: Any,
    ) -> Any:
        return latents

    def evaluate_candidate(
        self, image: Image.Image, context: GenerationContext
    ) -> CandidateDecision:
        return CandidateDecision()


class NoOpGuidance(GuidanceHook):
    pass


def load_guidance_hook(spec: str | None, config_path: Path | None = None) -> GuidanceHook:
    if not spec:
        return NoOpGuidance()
    if ":" not in spec:
        raise ValueError("Guidance hook must use module:factory syntax")
    module_name, factory_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    config: dict[str, Any] = {}
    if config_path is not None:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    hook = factory(config)
    if not isinstance(hook, GuidanceHook):
        raise TypeError(f"{spec} did not create a GuidanceHook")
    return hook


def apply_retry_feedback(
    context: GenerationContext, decision: CandidateDecision, retry_seed: int
) -> GenerationContext:
    morphology = context.morphology.copy()
    if decision.next_morphology_delta is not None:
        delta = np.asarray(decision.next_morphology_delta, dtype=np.float32)
        if delta.shape != morphology.shape:
            raise ValueError(
                f"Guidance morphology delta has shape {delta.shape}; expected {morphology.shape}"
            )
        morphology = morphology + delta
    metadata = dict(context.metadata)
    if decision.next_spatial_scale is not None:
        metadata["guidance_spatial_scale"] = float(decision.next_spatial_scale)
    metadata.setdefault("rejections", []).append(
        {"attempt": context.attempt, "score": decision.score, "reason": decision.reason}
    )
    return context.clone(
        morphology=morphology,
        seed=retry_seed,
        attempt=context.attempt + 1,
        metadata=metadata,
    )
