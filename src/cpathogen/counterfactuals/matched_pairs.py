"""Serializable records for baseline/counterfactual pairs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class MatchedPairRecord:
    """Provenance for two images generated from the same initial noise."""

    candidate_id: str
    stem: str
    seed: int
    prompt: str
    baseline_image: str | None
    counterfactual_image: str
    reference_tile: str | None
    intervention: dict[str, Any]
    applied_details: dict[str, Any] = field(default_factory=dict)
    difference: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
