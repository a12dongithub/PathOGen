from __future__ import annotations

import random
import unittest

import numpy as np
import torch

from cpathogen.counterfactuals.centroids import (
    add_jittered_centroids,
    remove_centroids,
    render_centroid_channel,
    sd_target_count,
)


class InflammatoryCentroidTests(unittest.TestCase):
    def test_sd_levels_produce_three_distinct_doses(self) -> None:
        self.assertEqual(
            [sd_target_count(10, level, 2.3037645) for level in (0.5, 1.0, 1.5)],
            [19, 30, 44],
        )

    def test_signed_sd_levels_produce_ordered_counts(self) -> None:
        levels = (-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0)
        counts = [sd_target_count(100, level, 2.3037645) for level in levels]
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(counts[2], 100)

    def test_negative_levels_remove_nested_deterministic_subsets(self) -> None:
        original = np.asarray([(index, index) for index in range(20)], dtype=np.int16)
        retained_one, removed_one = remove_centroids(
            original, 4, rng=random.Random(1729)
        )
        retained_two, removed_two = remove_centroids(
            original, 9, rng=random.Random(1729)
        )
        self.assertTrue(np.array_equal(removed_one, removed_two[:4]))
        self.assertTrue(
            set(map(tuple, retained_two)).issubset(set(map(tuple, retained_one)))
        )

    def test_dose_levels_are_nested_and_deterministic(self) -> None:
        original = np.asarray([[100, 100], [200, 200], [300, 300]], dtype=np.int16)
        results = []
        for count in (1, 2, 3):
            _, added = add_jittered_centroids(original, count, rng=random.Random(1729))
            results.append(added)
        self.assertTrue(np.array_equal(results[0], results[1][:1]))
        self.assertTrue(np.array_equal(results[1], results[2][:2]))

    def test_render_adds_new_spatial_peaks(self) -> None:
        original = np.asarray([[100, 100], [200, 200], [300, 300]], dtype=np.int16)
        combined, _ = add_jittered_centroids(original, 3, rng=random.Random(42))
        baseline = render_centroid_channel(original)
        converted = render_centroid_channel(combined)
        self.assertEqual(tuple(baseline.shape), (512, 512))
        self.assertFalse(torch.equal(baseline, converted))
        self.assertGreater(int((converted != baseline).sum()), 0)


if __name__ == "__main__":
    unittest.main()
