import csv
import importlib.util
from pathlib import Path


def load_cloud_run():
    path = (
        Path(__file__).parents[1]
        / "workflows"
        / "07_train_evaluate_probe"
        / "cloud_run.py"
    )
    spec = importlib.util.spec_from_file_location("probe_cloud_run", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_counterfactual_counts_accepts_four_matched_conditions(tmp_path: Path) -> None:
    module = load_cloud_run()
    manifest = tmp_path / "images.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id", "condition"])
        writer.writeheader()
        for candidate in ("candidate_0000", "candidate_0001"):
            for condition in ("baseline", "low", "medium", "high"):
                writer.writerow(
                    {"candidate_id": candidate, "condition": condition}
                )
    assert module.counterfactual_counts(tmp_path) == (8, 2)


def test_dataset_root_supports_archives_with_a_package_directory(
    tmp_path: Path,
) -> None:
    module = load_cloud_run()
    package = tmp_path / "package"
    package.mkdir()
    (package / "images.csv").write_text("candidate_id,condition\n", encoding="utf-8")
    assert module.dataset_root(tmp_path, "images.csv") == package
