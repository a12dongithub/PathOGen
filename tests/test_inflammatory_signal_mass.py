from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd
import torch

from cpathogen.counterfactuals import (
    CandidateRecord,
    ConditionBundle,
    load_candidate_manifest,
    select_candidate_shard,
)
from experiments.spatial.inflammatory_signal_mass import (
    INFLAMMATORY_CHANNEL,
    _increase_clamped_mass,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _cloud_run_module():
    path = REPOSITORY_ROOT / "workflows/05_generate_counterfactuals/cloud_run.py"
    spec = spec_from_file_location("cpathogen_cloud_run", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load cloud runner: {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CandidateManifestTests(unittest.TestCase):
    def test_manifest_preserves_explicit_stem_seed_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidates.csv"
            pd.DataFrame(
                {
                    "candidate_id": ["a", "b"],
                    "stem": ["tile-a", "tile-b"],
                    "seed": [11, 29],
                }
            ).to_csv(path, index=False)
            records = load_candidate_manifest(
                path, available_stems={"tile-a", "tile-b"}
            )
        self.assertEqual(
            records,
            [CandidateRecord("a", "tile-a", 11), CandidateRecord("b", "tile-b", 29)],
        )

    def test_interleaved_shards_cover_candidates_once(self) -> None:
        candidates = [CandidateRecord(str(i), f"tile-{i}", i) for i in range(10)]
        shards = [
            select_candidate_shard(candidates, shard_index=i, num_shards=3)
            for i in range(3)
        ]
        flattened = [item.candidate_id for shard in shards for item in shard]
        self.assertEqual(sorted(flattened, key=int), [str(i) for i in range(10)])


class InflammatoryMassTests(unittest.TestCase):
    def test_mass_increases_and_other_controls_are_unchanged(self) -> None:
        spatial = torch.rand(5, 12, 12) * 0.7
        morphology = torch.randn(16)
        original = ConditionBundle("tile", spatial, morphology)
        converted_channel, details = _increase_clamped_mass(
            original.spatial[INFLAMMATORY_CHANNEL], 0.30
        )
        converted_spatial = original.spatial.clone()
        converted_spatial[INFLAMMATORY_CHANNEL] = converted_channel
        self.assertAlmostEqual(details["achieved_fraction"], 0.30, places=5)
        self.assertTrue(
            torch.equal(converted_spatial[0], original.spatial[0])
            and torch.equal(converted_spatial[2:], original.spatial[2:])
        )
        self.assertTrue(torch.equal(original.morphology, morphology))

    def test_saturated_signal_is_reported_as_clipped(self) -> None:
        converted, details = _increase_clamped_mass(torch.ones(4, 4), 0.30)
        self.assertTrue(torch.equal(converted, torch.ones(4, 4)))
        self.assertTrue(details["target_clipped"])
        self.assertEqual(details["achieved_fraction"], 0.0)


class ProgressUploaderTests(unittest.TestCase):
    def test_concurrent_status_writes_are_atomic(self) -> None:
        cloud_run = _cloud_run_module()
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            uploader = cloud_run.ProgressUploader(
                workspace=workspace,
                outputs=workspace / "outputs",
                output_uri="gs://unused/test",
                interval_seconds=60,
            )
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda _: uploader._write_status(), range(100)))
            status = pd.read_json(uploader.status_path, typ="series")
        self.assertEqual(status["phase"], "starting")
        self.assertEqual(status["generated_png_count"], 0)


if __name__ == "__main__":
    unittest.main()
