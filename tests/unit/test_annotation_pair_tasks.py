from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cpathogen.annotation.cellvit_adapter import _pair_tasks


class AnnotationPairTaskTest(unittest.TestCase):
    def test_baseline_is_deduplicated_and_counterfactuals_keep_pair_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs_path = root / "pairs.jsonl"
            records = []
            for slug in ("first", "second"):
                records.append(
                    {
                        "stem": "tile-a",
                        "seed": 42,
                        "baseline_image": str(root / "baseline.png"),
                        "counterfactual_image": str(root / f"{slug}.png"),
                        "intervention": {"slug": slug, "name": slug},
                    }
                )
            pairs_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            tasks = _pair_tasks(pairs_path, root / "annotations")

        self.assertEqual(len(tasks), 3)
        baseline = next(task for task in tasks if task.source_metadata["pair_role"] == "baseline")
        counterfactuals = [
            task
            for task in tasks
            if task.source_metadata["pair_role"] == "counterfactual"
        ]
        self.assertNotIn("pair_id", baseline.source_metadata)
        self.assertEqual(
            baseline.source_metadata["source_kind"], "generated_baseline"
        )
        self.assertEqual(
            baseline.source_metadata["pair_group_id"], "tile-a:seed=42"
        )
        self.assertEqual(
            {task.source_metadata["pair_id"] for task in counterfactuals},
            {
                "tile-a:seed=42:intervention=first",
                "tile-a:seed=42:intervention=second",
            },
        )
        self.assertEqual(
            {task.source_metadata["source_kind"] for task in counterfactuals},
            {"generated_counterfactual"},
        )


if __name__ == "__main__":
    unittest.main()
