from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


REQUIRED_PATHS = [
    "01_one_step_baseline_pipeline.ipynb",
    "02_two_step_pipeline.ipynb",
    "03_model_comparison.ipynb",
    "prepared_dataset/images/train",
    "prepared_dataset/labels/train",
    "prepared_dataset/images/test",
    "prepared_dataset/labels/test",
    "runs/comparison/comparison_summary.csv",
    "runs/two_step/evaluation/two_step_match_details.csv",
    "scripts/evaluate_conditions.py",
    "scripts/live_fall_gui.py",
    "scripts/run_gui.py",
    "scripts/run_report_assets.py",
]


def count_files(path: Path, suffixes: set[str] | None = None) -> int:
    if not path.exists():
        return 0
    files = [item for item in path.rglob("*") if item.is_file()]
    if suffixes is not None:
        files = [item for item in files if item.suffix.lower() in suffixes]
    return len(files)


def main() -> None:
    failures = []
    print("Project readiness check")
    print("=" * 24)

    for relative_path in REQUIRED_PATHS:
        path = PROJECT_ROOT / relative_path
        status = "OK" if path.exists() else "MISSING"
        print(f"{status:7} {relative_path}")
        if not path.exists():
            failures.append(relative_path)

    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    dataset_counts = {
        "train images": count_files(PROJECT_ROOT / "prepared_dataset/images/train", image_suffixes),
        "train labels": count_files(PROJECT_ROOT / "prepared_dataset/labels/train", {".txt"}),
        "test images": count_files(PROJECT_ROOT / "prepared_dataset/images/test", image_suffixes),
        "test labels": count_files(PROJECT_ROOT / "prepared_dataset/labels/test", {".txt"}),
        "raw test images": count_files(PROJECT_ROOT / "test_dataset", image_suffixes),
    }

    print()
    print("Dataset counts")
    print("=" * 14)
    for label, count in dataset_counts.items():
        print(f"{label:16} {count}")

    if dataset_counts["train images"] != dataset_counts["train labels"]:
        failures.append("train image/label count mismatch")
    if dataset_counts["test images"] != dataset_counts["test labels"]:
        failures.append("test image/label count mismatch")

    report_assets = PROJECT_ROOT / "runs/report_assets"
    expected_assets = [
        "condition_summary.csv",
        "condition_per_class.csv",
        "condition_f1_by_lighting.png",
        "report_asset_notes.json",
    ]
    print()
    print("Report assets")
    print("=" * 13)
    for asset in expected_assets:
        path = report_assets / asset
        status = "OK" if path.exists() else "MISSING"
        print(f"{status:7} runs/report_assets/{asset}")
        if not path.exists():
            failures.append(f"runs/report_assets/{asset}")

    print()
    if failures:
        print("Readiness check failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("Readiness check passed.")


if __name__ == "__main__":
    main()
