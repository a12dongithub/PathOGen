from __future__ import annotations

import unittest

import torch

from cpathogen.counterfactuals import (
    ConditionBundle,
    IdentityIntervention,
    InterventionContext,
    MatchedPairRecord,
    load_interventions,
    select_interventions,
)


class _FakeStore:
    stems = ("source", "donor-a", "donor-b")

    def load_spatial(self, stem: str) -> torch.Tensor:
        value = float(self.stems.index(stem)) / 2.0
        return torch.full((5, 8, 8), value)

    def load_morphology(self, stem: str) -> torch.Tensor:
        return torch.full((16,), float(self.stems.index(stem)))


class CounterfactualContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = _FakeStore()
        spatial = torch.zeros(5, 8, 8)
        spatial[0, 2:6, 2:6] = 1.0
        self.original = ConditionBundle("source", spatial, torch.zeros(16))
        self.context = InterventionContext(self.store, "source", 42, 99)  # type: ignore[arg-type]

    def test_identity_returns_equal_but_independent_tensors(self) -> None:
        result = IdentityIntervention().apply(self.original, self.context).condition
        self.assertTrue(torch.equal(result.spatial, self.original.spatial))
        self.assertTrue(torch.equal(result.morphology, self.original.morphology))
        self.assertNotEqual(result.spatial.data_ptr(), self.original.spatial.data_ptr())
        self.assertNotEqual(
            result.morphology.data_ptr(), self.original.morphology.data_ptr()
        )

    def test_experiment_loader_and_selection(self) -> None:
        _, interventions = load_interventions(
            "experiments.spatial.relabel_all_cells"
        )
        selected = select_interventions(interventions, ["all_cells_inflammatory"])
        self.assertEqual([item.slug for item in selected], ["all_cells_inflammatory"])
        converted = selected[0].apply(self.original, self.context).condition
        self.assertEqual(float(converted.spatial[1].max()), 1.0)
        self.assertEqual(float(converted.spatial[[0, 2, 3, 4]].max()), 0.0)
        self.assertTrue(torch.equal(converted.morphology, self.original.morphology))

    def test_donor_choice_is_reproducible_and_excludes_source(self) -> None:
        first = self.context.donor_stem("test")
        second = self.context.donor_stem("test")
        self.assertEqual(first, second)
        self.assertNotEqual(first, "source")

    def test_pair_record_is_serializable(self) -> None:
        record = MatchedPairRecord(
            stem="source",
            seed=99,
            prompt="he",
            baseline_image="baseline.png",
            counterfactual_image="counterfactual.png",
            reference_tile=None,
            intervention={"slug": "identity"},
        )
        self.assertEqual(record.to_dict()["seed"], 99)


if __name__ == "__main__":
    unittest.main()
