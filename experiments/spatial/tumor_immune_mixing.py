"""Redistribute inflammatory signal toward tumor-rich regions at fixed mass."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor

from cpathogen.counterfactuals import ConditionIntervention, InterventionContext

TUMOR_CHANNEL = 0
INFLAMMATORY_CHANNEL = 1
TARGET_BLUR_SIGMA_PX = 12.0
TARGET_BLUR_RADIUS_PX = 36


def _scale_to_capped_mass(weights: Tensor, target_mass: float) -> Tensor:
    """Scale nonnegative weights into [0, 1] with the requested total mass."""
    values = weights.to(dtype=torch.float64).clamp_min(0.0)
    if not 0.0 < target_mass <= values.numel():
        raise ValueError("Target mass must be in (0, number of pixels]")
    if float(values.sum()) <= 0.0:
        raise ValueError("Mixing weights must have positive mass")
    low, high = 0.0, 1.0
    while float(torch.clamp(values * high, max=1.0).sum()) < target_mass:
        high *= 2.0
    for _ in range(64):
        midpoint = (low + high) / 2.0
        if float(torch.clamp(values * midpoint, max=1.0).sum()) < target_mass:
            low = midpoint
        else:
            high = midpoint
    return torch.clamp(values * ((low + high) / 2.0), max=1.0)


def _maximally_tumor_centered(tumor: Tensor, target_mass: float) -> Tensor:
    """Fallback target that maximizes tumor-weighted inflammatory overlap."""
    flat = torch.zeros(tumor.numel(), dtype=torch.float64, device=tumor.device)
    order = torch.argsort(tumor.flatten(), descending=True)
    whole = min(int(target_mass), len(order))
    flat[order[:whole]] = 1.0
    remainder = target_mass - whole
    if remainder > 0.0 and whole < len(order):
        flat[order[whole]] = remainder
    return flat.reshape(tumor.shape)


def tumor_weighted_overlap(tumor: Tensor, inflammatory: Tensor) -> float:
    mass = float(inflammatory.sum())
    if mass <= 0.0:
        raise ValueError("Inflammatory channel has no mass")
    return float((tumor * inflammatory).sum()) / mass


def tumor_attracted_target(tumor: Tensor, inflammatory: Tensor) -> Tensor:
    """Construct a smooth inflammatory target centered on tumor regions."""
    tumor64 = tumor.to(dtype=torch.float64)
    immune64 = inflammatory.to(dtype=torch.float64)
    mass = float(immune64.sum())
    coordinates = torch.arange(
        -TARGET_BLUR_RADIUS_PX,
        TARGET_BLUR_RADIUS_PX + 1,
        dtype=torch.float64,
        device=tumor.device,
    )
    kernel_1d = torch.exp(-0.5 * (coordinates / TARGET_BLUR_SIGMA_PX) ** 2)
    kernel_1d /= kernel_1d.sum()
    kernel = torch.outer(kernel_1d, kernel_1d)[None, None]
    blurred_tumor = functional.conv2d(
        tumor64[None, None], kernel, padding=TARGET_BLUR_RADIUS_PX
    )[0, 0]
    epsilon = max(float(blurred_tumor.mean()) * 1e-6, 1e-12)
    weights = blurred_tumor + epsilon
    target = _scale_to_capped_mass(weights, mass)
    if tumor_weighted_overlap(tumor64, target) <= tumor_weighted_overlap(
        tumor64, immune64
    ):
        target = _maximally_tumor_centered(tumor64, mass)
    return target.to(dtype=inflammatory.dtype)


class TumorImmuneMixing(ConditionIntervention):
    """Interpolate inflammatory signal toward a tumor-attracted target."""

    def __init__(self, mixing_fraction: float) -> None:
        self.mixing_fraction = float(mixing_fraction)
        if not 0.0 < self.mixing_fraction < 1.0:
            raise ValueError("mixing_fraction must be between zero and one")
        label = str(self.mixing_fraction).replace(".", "p")
        self.name = f"tumor_immune_mixing_{label}"

    def parameters(self) -> dict[str, Any]:
        return {
            "mixing_fraction": self.mixing_fraction,
            "preserved_quantity": "inflammatory channel mass",
            "target": "Gaussian-smoothed tumor-centered inflammatory distribution",
            "target_blur_sigma_px": TARGET_BLUR_SIGMA_PX,
        }

    def modify_spatial(
        self, spatial: Tensor, context: InterventionContext
    ) -> Tensor:
        del context
        tumor = spatial[TUMOR_CHANNEL]
        inflammatory = spatial[INFLAMMATORY_CHANNEL]
        target = tumor_attracted_target(tumor, inflammatory)
        spatial[INFLAMMATORY_CHANNEL] = (
            (1.0 - self.mixing_fraction) * inflammatory
            + self.mixing_fraction * target
        )
        return spatial

    def details(self, context: InterventionContext) -> dict[str, Any]:
        original = context.store.load_spatial(context.original_stem)
        tumor = original[TUMOR_CHANNEL]
        inflammatory = original[INFLAMMATORY_CHANNEL]
        target = tumor_attracted_target(tumor, inflammatory)
        converted = (
            (1.0 - self.mixing_fraction) * inflammatory
            + self.mixing_fraction * target
        )
        return {
            "mixing_fraction": self.mixing_fraction,
            "inflammatory_mass_before": float(inflammatory.sum()),
            "inflammatory_mass_after": float(converted.sum()),
            "tumor_weighted_overlap_before": tumor_weighted_overlap(
                tumor, inflammatory
            ),
            "tumor_weighted_overlap_after": tumor_weighted_overlap(tumor, converted),
            "changed_spatial_channel": "inflammatory",
        }


def build_interventions() -> list[ConditionIntervention]:
    return [TumorImmuneMixing(0.25), TumorImmuneMixing(0.50), TumorImmuneMixing(0.75)]
