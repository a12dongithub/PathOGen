"""Manifest-driven candidate selection for matched counterfactual generation."""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class CandidateRecord:
    """One selected condition and its one selected diffusion seed."""

    candidate_id: str
    stem: str
    seed: int


def load_candidate_manifest(
    path: str | Path,
    *,
    available_stems: Collection[str] | None = None,
) -> list[CandidateRecord]:
    """Load a canonical CSV without constructing a stem-by-seed product."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Candidate manifest not found: {path}")
    table = pd.read_csv(path)
    required = {"candidate_id", "stem", "seed"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Candidate manifest is missing columns: {missing}")
    if table.empty:
        raise ValueError("Candidate manifest contains no rows")
    if table["candidate_id"].isna().any() or table["stem"].isna().any():
        raise ValueError("Candidate IDs and stems cannot be missing")

    candidate_ids = table["candidate_id"].astype(str)
    stems = table["stem"].astype(str)
    if not candidate_ids.is_unique:
        duplicates = sorted(candidate_ids[candidate_ids.duplicated()].unique())
        raise ValueError(f"Candidate IDs must be unique: {duplicates[:10]}")
    unsafe = [value for value in candidate_ids if not _SAFE_ID.fullmatch(value)]
    if unsafe:
        raise ValueError(
            "Candidate IDs may contain only letters, numbers, '.', '_' and '-': "
            f"{unsafe[:10]}"
        )

    numeric_seeds = pd.to_numeric(table["seed"], errors="coerce")
    if numeric_seeds.isna().any() or (numeric_seeds % 1 != 0).any():
        raise ValueError("Every candidate seed must be an integer")
    seeds = numeric_seeds.astype("int64")
    if (seeds < 0).any():
        raise ValueError("Candidate seeds must be non-negative")

    if available_stems is not None:
        unavailable = sorted(set(stems).difference(available_stems))
        if unavailable:
            raise KeyError(
                "Candidate stems are not aligned in the condition store: "
                f"{unavailable[:10]}"
            )

    return [
        CandidateRecord(candidate_id, stem, int(seed))
        for candidate_id, stem, seed in zip(candidate_ids, stems, seeds, strict=True)
    ]


def select_candidate_shard(
    candidates: list[CandidateRecord], *, shard_index: int, num_shards: int
) -> list[CandidateRecord]:
    """Select one deterministic, interleaved shard from an ordered manifest."""
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    selected = candidates[shard_index::num_shards]
    if not selected:
        raise ValueError(
            f"Shard {shard_index} is empty for {len(candidates)} candidates and "
            f"{num_shards} shards"
        )
    return selected
