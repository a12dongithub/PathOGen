"""Composable, in-memory transformations of Phase-2 control tensors."""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Any

from torch import Tensor

from .conditions import ConditionBundle, ConditionStore


def safe_slug(value: str) -> str:
    """Return a filesystem-safe, stable identifier."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("._-").lower()
    if not slug:
        raise ValueError("Intervention name cannot be empty")
    return slug


@dataclass(frozen=True)
class InterventionContext:
    """Read-only resources and seeds available to one intervention application."""

    store: ConditionStore
    original_stem: str
    intervention_seed: int
    generation_seed: int

    def rng(self, namespace: str) -> random.Random:
        payload = (
            f"{self.intervention_seed}|{self.original_stem}|{namespace}".encode("utf-8")
        )
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        return random.Random(seed)

    def donor_stem(self, namespace: str) -> str:
        candidates = self.store.stems
        if len(candidates) < 2:
            raise ValueError("A donor intervention requires at least two aligned tiles")
        rng = self.rng(namespace)
        donor = candidates[rng.randrange(len(candidates))]
        while donor == self.original_stem:
            donor = candidates[rng.randrange(len(candidates))]
        return donor


@dataclass(frozen=True)
class AppliedIntervention:
    """A transformed condition plus JSON-serializable provenance."""

    condition: ConditionBundle
    details: dict[str, Any] = field(default_factory=dict)


class ConditionIntervention:
    """Base class for experiment-defined control transformations.

    The default methods are identities. Experiments override only the spatial or
    morphology path they need; the workflow owns all data loading and inference.
    """

    name = "identity"

    @property
    def slug(self) -> str:
        return safe_slug(self.name)

    def parameters(self) -> dict[str, Any]:
        return {}

    def modify_spatial(
        self, spatial: Tensor, context: InterventionContext
    ) -> Tensor:
        return spatial

    def modify_morphology(
        self, morphology: Tensor, context: InterventionContext
    ) -> Tensor:
        return morphology

    def details(self, context: InterventionContext) -> dict[str, Any]:
        return {}

    def apply(
        self, original: ConditionBundle, context: InterventionContext
    ) -> AppliedIntervention:
        if original.stem != context.original_stem:
            raise ValueError("Intervention context does not match the condition stem")
        spatial = self.modify_spatial(original.spatial.detach().clone(), context)
        morphology = self.modify_morphology(
            original.morphology.detach().clone(), context
        )
        converted = ConditionBundle(
            stem=original.stem,
            spatial=spatial,
            morphology=morphology,
            metadata={
                **original.metadata,
                "intervention": self.slug,
                "intervention_parameters": self.parameters(),
            },
        )
        converted.validate()
        return AppliedIntervention(converted, self.details(context))

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "slug": self.slug,
            "class": f"{type(self).__module__}.{type(self).__qualname__}",
            "parameters": self.parameters(),
        }


class IdentityIntervention(ConditionIntervention):
    """Explicit identity transform, useful for API and unit tests."""

    name = "identity"
