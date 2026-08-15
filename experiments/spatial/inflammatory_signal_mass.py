"""Prespecified +10%, +20%, and +30% inflammatory-signal mass sweep."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from cpathogen.counterfactuals import CELL_TYPE_NAMES
from cpathogen.counterfactuals.interventions import (
    ConditionIntervention,
    InterventionContext,
)

INFLAMMATORY_CHANNEL = CELL_TYPE_NAMES.index("inflammatory")


def _increase_clamped_mass(
    channel: Tensor, fraction: float
) -> tuple[Tensor, dict[str, Any]]:
    if fraction <= 0:
        raise ValueError("Inflammatory mass increase must be positive")
    original_mass = float(channel.sum(dtype=torch.float64))
    requested_mass = original_mass * (1.0 + fraction)
    maximum_mass = float((channel > 0).sum())
    target_mass = min(requested_mass, maximum_mass)

    if original_mass == 0.0 or target_mass == original_mass:
        multiplier = 1.0
        converted = channel.clone()
    else:
        low = 1.0
        high = 1.0 + fraction
        while (
            float(torch.clamp(channel * high, 0.0, 1.0).sum(dtype=torch.float64))
            < target_mass
        ):
            high *= 2.0
        for _ in range(64):
            midpoint = (low + high) / 2.0
            mass = float(
                torch.clamp(channel * midpoint, 0.0, 1.0).sum(dtype=torch.float64)
            )
            if mass < target_mass:
                low = midpoint
            else:
                high = midpoint
        multiplier = high
        converted = torch.clamp(channel * multiplier, 0.0, 1.0)

    achieved_mass = float(converted.sum(dtype=torch.float64))
    achieved_fraction = (
        achieved_mass / original_mass - 1.0 if original_mass > 0.0 else 0.0
    )
    return converted, {
        "channel": "inflammatory",
        "channel_index": INFLAMMATORY_CHANNEL,
        "requested_fraction": fraction,
        "original_mass": original_mass,
        "requested_mass": requested_mass,
        "target_mass_after_feasibility_clip": target_mass,
        "achieved_mass": achieved_mass,
        "achieved_fraction": achieved_fraction,
        "multiplier": multiplier,
        "target_clipped": target_mass < requested_mass,
    }


class InflammatorySignalMassIncrease(ConditionIntervention):
    """Increase inflammatory signal mass while preserving its spatial pattern."""

    def __init__(self, fraction: float) -> None:
        self.fraction = float(fraction)
        if self.fraction <= 0:
            raise ValueError("fraction must be positive")
        self.name = f"inflammatory_mass_{round(self.fraction * 100):02d}pct"

    def parameters(self) -> dict[str, Any]:
        return {"fraction": self.fraction, "channel": "inflammatory"}

    def modify_spatial(self, spatial: Tensor, context: InterventionContext) -> Tensor:
        converted, _ = _increase_clamped_mass(
            spatial[INFLAMMATORY_CHANNEL], self.fraction
        )
        spatial[INFLAMMATORY_CHANNEL] = converted
        return spatial

    def details(self, context: InterventionContext) -> dict[str, Any]:
        original = context.store.load_spatial(context.original_stem)
        _, details = _increase_clamped_mass(
            original[INFLAMMATORY_CHANNEL], self.fraction
        )
        return details


def build_interventions() -> list[ConditionIntervention]:
    return [
        InflammatorySignalMassIncrease(0.10),
        InflammatorySignalMassIncrease(0.20),
        InflammatorySignalMassIncrease(0.30),
    ]
