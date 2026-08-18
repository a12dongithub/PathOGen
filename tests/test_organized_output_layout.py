import importlib.util
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[1]


def load_module(name: str, relative_path: str):
    path = REPOSITORY_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tile_folder_layout_uses_stem_and_descriptive_condition() -> None:
    runner = load_module(
        "counterfactual_runner_layout_test",
        "workflows/05_generate_counterfactuals/run.py",
    )
    candidate = runner.CandidateRecord(
        candidate_id="candidate_0007",
        stem="TCGA-A1-A0SB_x77824_y47104_BR",
        seed=843410356,
    )
    root = Path("output")
    assert runner._candidate_image_dir(
        root, candidate, tile_folder_layout=True
    ) == root / candidate.stem
    assert runner._candidate_image_dir(
        root, candidate, tile_folder_layout=False
    ) == root / "images" / candidate.candidate_id / "seed_0843410356"


def test_tile_folder_layout_rejects_path_traversal() -> None:
    runner = load_module(
        "counterfactual_runner_safety_test",
        "workflows/05_generate_counterfactuals/run.py",
    )
    candidate = runner.CandidateRecord("candidate_0000", "../escape", 42)
    with pytest.raises(ValueError, match="Unsafe tile stem"):
        runner._candidate_image_dir(
            Path("output"), candidate, tile_folder_layout=True
        )


def test_bucket_manifest_mapping_uses_stem_and_condition() -> None:
    organizer = load_module(
        "bucket_organizer_mapping_test",
        "workflows/05_generate_counterfactuals/organize_bucket_experiments.py",
    )
    source = organizer.ManifestSource("nuclear_enlargement", "source/run")
    items = organizer.plan_from_manifest_rows(
        source,
        [
            {
                "candidate_id": "candidate_0000",
                "stem": "TCGA-A1-A0SB_x77824_y47104_BR",
                "seed": "843410356",
                "condition": "nuclear_enlargement_plus_1p0sd",
                "image_path": (
                    "/workspace/images/candidate_0000/seed_0843410356/"
                    "nuclear_enlargement_plus_1p0sd.png"
                ),
            }
        ],
    )
    assert len(items) == 1
    item = items[0]
    assert item.source_object == (
        "source/run/images/candidate_0000/seed_0843410356/"
        "nuclear_enlargement_plus_1p0sd.png"
    )
    assert item.relative_destination == (
        "nuclear_enlargement/TCGA-A1-A0SB_x77824_y47104_BR/"
        "nuclear_enlargement_plus_1p0sd.png"
    )


def test_complete_three_experiment_plan_validation() -> None:
    organizer = load_module(
        "bucket_organizer_validation_test",
        "workflows/05_generate_counterfactuals/organize_bucket_experiments.py",
    )
    items = []
    panel_sets = {
        "nuclear_enlargement": [
            ("complete", organizer.EXPECTED_CONDITIONS["nuclear_enlargement"], 770),
            ("base", organizer.BASE_CONDITIONS["nuclear_enlargement"], 230),
            (
                "extension",
                organizer.EXTENSION_CONDITIONS["nuclear_enlargement"],
                230,
            ),
        ],
        "nuclear_shape_irregularity": [
            (
                "complete",
                organizer.EXPECTED_CONDITIONS["nuclear_shape_irregularity"],
                1000,
            )
        ],
        "stain_brightness": [
            ("complete", organizer.EXPECTED_CONDITIONS["stain_brightness"], 770),
            ("base", organizer.BASE_CONDITIONS["stain_brightness"], 230),
            (
                "extension",
                organizer.EXTENSION_CONDITIONS["stain_brightness"],
                230,
            ),
        ],
    }
    for experiment, cohorts in panel_sets.items():
        candidate_index = 0
        for cohort_name, conditions, count in cohorts:
            for index in range(count):
                stem = f"tile_{cohort_name}_{index:04d}"
                for condition in conditions:
                    relative = f"{experiment}/{stem}/{condition}.png"
                    items.append(
                        organizer.OrganizedImage(
                            experiment=experiment,
                            candidate_id=f"candidate_{candidate_index:04d}",
                            stem=stem,
                            seed=candidate_index,
                            condition=condition,
                            source_object=f"source/{relative}",
                            relative_destination=relative,
                        )
                    )
                candidate_index += 1
    organizer.validate_plan(items)
    summary = organizer.panel_summary(items)
    assert summary["nuclear_enlargement"] == {
        "images": 7000,
        "tile_folders": 1230,
        "complete_panels": 770,
        "partial_panels": 460,
    }
