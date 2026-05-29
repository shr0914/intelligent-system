from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = {
    0: "fall detected",
    1: "walk",
    2: "sit",
}
FOLDER_CLASS_TO_ID = {
    "Fall": 0,
    "Walk": 1,
    "Sit": 2,
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
CHALLENGING_LIGHTING_VALUES = {"low_light", "dim", "shadowy"}


def normalize_condition_value(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def is_challenging_lighting(value: object) -> bool:
    return normalize_condition_value(value) in CHALLENGING_LIGHTING_VALUES


def select_challenging_lighting(metadata: pd.DataFrame) -> pd.DataFrame:
    if "lighting" not in metadata.columns:
        return metadata.iloc[0:0].copy()
    return metadata[metadata["lighting"].map(is_challenging_lighting)].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate condition-based metrics and report-ready assets."
    )
    parser.add_argument("--prepared", type=Path, default=PROJECT_ROOT / "prepared_dataset")
    parser.add_argument("--raw-test", type=Path, default=PROJECT_ROOT / "test_dataset")
    parser.add_argument(
        "--one-step-predictions",
        type=Path,
        default=None,
        help="Path to Ultralytics predictions.json. Defaults to newest available file.",
    )
    parser.add_argument(
        "--two-step-matches",
        type=Path,
        default=PROJECT_ROOT / "runs" / "two_step" / "evaluation" / "two_step_match_details.csv",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "report_assets")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--one-step-conf",
        type=float,
        default=0.25,
        help="Minimum confidence for one-step JSON predictions used in fixed-threshold condition metrics.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=(
            "Optional CSV with image metadata. Supported columns: image_path, "
            "file_name, stem, lighting, viewpoint, distance, environment."
        ),
    )
    parser.add_argument(
        "--condition-distribution",
        type=Path,
        default=PROJECT_ROOT / "Test Dataset Distribution",
        help=(
            "Optional folder with condition subfolders. Expected layout: "
            "lighting/<value>/*.jpeg, environment/<value>/*.jpeg, "
            "viewpoint/<value>/*.jpeg, distance/<value>/*.jpeg. "
            "If present, it overrides matching columns from --metadata."
        ),
    )
    return parser.parse_args()


def find_one_step_predictions() -> Path:
    candidates = sorted(
        (PROJECT_ROOT / "runs" / "evaluation").glob("one_step_test_eval*/predictions.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("Could not find one-step predictions.json under runs/evaluation")
    return candidates[0]


def build_raw_metadata(raw_root: Path) -> pd.DataFrame:
    rows = []
    for image_path in sorted(raw_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = image_path.relative_to(raw_root)
        lighting = relative.parts[0] if len(relative.parts) >= 1 else "unknown"
        folder_class = relative.parts[1] if len(relative.parts) >= 2 else "unknown"
        rows.append(
            {
                "file_name": image_path.name,
                "stem": image_path.stem,
                "raw_path": str(image_path),
                "lighting": lighting,
                "viewpoint": "",
                "distance": "",
                "environment": "",
                "folder_class": folder_class,
                "folder_class_id": FOLDER_CLASS_TO_ID.get(folder_class),
            }
        )
    return pd.DataFrame(rows)


def load_manual_metadata(metadata_path: Path | None) -> pd.DataFrame:
    if metadata_path is None or not metadata_path.exists():
        return pd.DataFrame()
    metadata = pd.read_csv(metadata_path)
    if "stem" not in metadata.columns:
        if "file_name" in metadata.columns:
            metadata["stem"] = metadata["file_name"].map(lambda value: Path(str(value)).stem)
        elif "image_path" in metadata.columns:
            metadata["stem"] = metadata["image_path"].map(lambda value: Path(str(value)).stem)
    return metadata


def build_prepared_metadata(prepared_root: Path, raw_metadata: pd.DataFrame) -> pd.DataFrame:
    raw_by_stem = raw_metadata.drop_duplicates("stem").set_index("stem")
    rows = []
    for image_path in sorted((prepared_root / "images" / "test").iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        raw = raw_by_stem.loc[image_path.stem] if image_path.stem in raw_by_stem.index else {}
        rows.append(
            {
                "image_path": str(image_path),
                "file_name": image_path.name,
                "stem": image_path.stem,
                "lighting": raw.get("lighting", "unknown") if isinstance(raw, pd.Series) else "unknown",
                "viewpoint": raw.get("viewpoint", "") if isinstance(raw, pd.Series) else "",
                "distance": raw.get("distance", "") if isinstance(raw, pd.Series) else "",
                "environment": raw.get("environment", "") if isinstance(raw, pd.Series) else "",
                "folder_class": raw.get("folder_class", "unknown") if isinstance(raw, pd.Series) else "unknown",
                "folder_class_id": raw.get("folder_class_id") if isinstance(raw, pd.Series) else None,
            }
        )
    return pd.DataFrame(rows)


def apply_manual_metadata(metadata: pd.DataFrame, manual_metadata: pd.DataFrame) -> pd.DataFrame:
    if manual_metadata.empty or "stem" not in manual_metadata.columns:
        return metadata

    manual_by_stem = manual_metadata.drop_duplicates("stem").set_index("stem")
    enriched = metadata.copy()
    for column in ["lighting", "viewpoint", "distance", "environment"]:
        if column not in enriched.columns:
            enriched[column] = ""
        if column not in manual_by_stem.columns:
            continue
        for index, row in enriched.iterrows():
            stem = row["stem"]
            if stem not in manual_by_stem.index:
                continue
            manual_value = manual_by_stem.loc[stem, column]
            if pd.notna(manual_value) and str(manual_value).strip():
                enriched.at[index, column] = str(manual_value).strip()
    return enriched


def apply_condition_distribution(metadata: pd.DataFrame, distribution_root: Path | None) -> pd.DataFrame:
    if distribution_root is None or not distribution_root.exists():
        return metadata

    group_to_column = {
        "lighting": "lighting",
        "environment": "environment",
        "viewpoint": "viewpoint",
        "distance": "distance",
    }
    prepared_names = set(metadata["file_name"].astype(str))
    enriched = metadata.copy()
    errors = []

    for group_name, column in group_to_column.items():
        group_dir = distribution_root / group_name
        if not group_dir.exists():
            continue

        mapping: dict[str, str] = {}
        for category_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            category = category_dir.name.strip()
            for image_path in sorted(path for path in category_dir.rglob("*") if path.is_file()):
                file_name = image_path.name
                if file_name in mapping:
                    errors.append(f"Duplicate {group_name} entry for {file_name}")
                mapping[file_name] = category

        missing = prepared_names - set(mapping)
        extra = set(mapping) - prepared_names
        if missing:
            sample = ", ".join(sorted(missing)[:5])
            errors.append(f"{group_name} distribution missing {len(missing)} prepared images. Sample: {sample}")
        if extra:
            sample = ", ".join(sorted(extra)[:5])
            errors.append(f"{group_name} distribution has {len(extra)} extra images. Sample: {sample}")

        enriched[column] = enriched["file_name"].map(mapping).fillna(enriched.get(column, ""))

    if errors:
        raise ValueError("\n".join(errors))

    return enriched


def yolo_to_xyxy(row: str, width: int, height: int) -> dict[str, float]:
    class_id, x_center, y_center, box_width, box_height = map(float, row.split()[:5])
    x1 = (x_center - box_width / 2) * width
    y1 = (y_center - box_height / 2) * height
    x2 = (x_center + box_width / 2) * width
    y2 = (y_center + box_height / 2) * height
    return {"class_id": int(class_id), "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def load_ground_truth(prepared_root: Path, metadata: pd.DataFrame) -> dict[str, list[dict[str, float]]]:
    gt_by_file = {}
    label_root = prepared_root / "labels" / "test"
    for row in metadata.itertuples(index=False):
        image = cv2.imread(row.image_path)
        if image is None:
            gt_by_file[row.file_name] = []
            continue
        height, width = image.shape[:2]
        label_path = label_root / f"{row.stem}.txt"
        boxes = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    boxes.append(yolo_to_xyxy(line, width, height))
        gt_by_file[row.file_name] = boxes
    return gt_by_file


def iou(box_a: dict[str, float], box_b: dict[str, float]) -> float:
    inter_x1 = max(box_a["x1"], box_b["x1"])
    inter_y1 = max(box_a["y1"], box_b["y1"])
    inter_x2 = min(box_a["x2"], box_b["x2"])
    inter_y2 = min(box_a["y2"], box_b["y2"])
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, box_a["x2"] - box_a["x1"]) * max(0.0, box_a["y2"] - box_a["y1"])
    area_b = max(0.0, box_b["x2"] - box_b["x1"]) * max(0.0, box_b["y2"] - box_b["y1"])
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def summarize_counts(
    records: list[dict[str, object]],
    condition_column: str = "lighting",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    class_rows = []
    df = pd.DataFrame(records)
    if condition_column not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    df[condition_column] = df[condition_column].fillna("").replace("", "unlabeled")
    for condition_value, group in df.groupby(condition_column, dropna=False):
        tp = int((group["match_type"] == "tp").sum())
        fp = int((group["match_type"] == "fp").sum())
        fn = int((group["match_type"] == "fn").sum())
        summary_rows.append(metric_row(str(condition_value), "all", tp, fp, fn, condition_column=condition_column))
        for class_id, class_group in group.groupby("class_id", dropna=False):
            tp = int((class_group["match_type"] == "tp").sum())
            fp = int((class_group["match_type"] == "fp").sum())
            fn = int((class_group["match_type"] == "fn").sum())
            class_rows.append(
                metric_row(
                    str(condition_value),
                    CLASS_NAMES.get(int(class_id), str(class_id)),
                    tp,
                    fp,
                    fn,
                    int(class_id),
                    condition_column=condition_column,
                )
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(class_rows)


def metric_row(
    condition_value: str,
    class_name: str,
    tp: int,
    fp: int,
    fn: int,
    class_id: int | None = None,
    condition_column: str = "lighting",
) -> dict[str, object]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        condition_column: condition_value,
        "class_id": class_id,
        "class_name": class_name,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_one_step(
    predictions_path: Path,
    metadata: pd.DataFrame,
    gt_by_file: dict[str, list[dict[str, float]]],
    iou_threshold: float,
    confidence_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata_by_file = metadata.set_index("file_name")
    with predictions_path.open(encoding="utf-8") as json_file:
        predictions = json.load(json_file)

    category_ids = {int(pred["category_id"]) for pred in predictions}
    one_indexed_categories = category_ids and category_ids.issubset({1, 2, 3}) and 0 not in category_ids

    preds_by_file: dict[str, list[dict[str, float]]] = defaultdict(list)
    for pred in predictions:
        confidence = float(pred.get("score", 0.0))
        if confidence < confidence_threshold:
            continue
        class_id = int(pred["category_id"]) - 1 if one_indexed_categories else int(pred["category_id"])
        if class_id not in CLASS_NAMES:
            continue
        x, y, width, height = pred["bbox"]
        preds_by_file[pred["file_name"]].append(
            {
                "class_id": class_id,
                "x1": float(x),
                "y1": float(y),
                "x2": float(x + width),
                "y2": float(y + height),
                "score": confidence,
            }
        )

    match_rows = []
    for file_name, gt_boxes in gt_by_file.items():
        metadata_row = metadata_by_file.loc[file_name] if file_name in metadata_by_file.index else {}
        lighting = metadata_row.get("lighting", "unknown") if isinstance(metadata_row, pd.Series) else "unknown"
        viewpoint = metadata_row.get("viewpoint", "") if isinstance(metadata_row, pd.Series) else ""
        distance = metadata_row.get("distance", "") if isinstance(metadata_row, pd.Series) else ""
        environment = metadata_row.get("environment", "") if isinstance(metadata_row, pd.Series) else ""
        used = [False] * len(gt_boxes)
        for pred in sorted(preds_by_file.get(file_name, []), key=lambda item: item["score"], reverse=True):
            best_iou = 0.0
            best_idx = None
            for idx, gt in enumerate(gt_boxes):
                if used[idx] or gt["class_id"] != pred["class_id"]:
                    continue
                score = iou(pred, gt)
                if score > best_iou:
                    best_iou = score
                    best_idx = idx
            if best_idx is not None and best_iou >= iou_threshold:
                used[best_idx] = True
                match_type = "tp"
            else:
                match_type = "fp"
            match_rows.append(
                {
                    "model": "one-step YOLO",
                    "file_name": file_name,
                    "lighting": lighting,
                    "viewpoint": viewpoint,
                    "distance": distance,
                    "environment": environment,
                    "class_id": pred["class_id"],
                    "class_name": CLASS_NAMES[pred["class_id"]],
                    "iou": best_iou,
                    "match_type": match_type,
                }
            )
        for idx, gt in enumerate(gt_boxes):
            if not used[idx]:
                match_rows.append(
                    {
                        "model": "one-step YOLO",
                        "file_name": file_name,
                        "lighting": lighting,
                        "viewpoint": viewpoint,
                        "distance": distance,
                        "environment": environment,
                        "class_id": gt["class_id"],
                        "class_name": CLASS_NAMES[gt["class_id"]],
                        "iou": None,
                        "match_type": "fn",
                    }
                )

    summary, per_class = summarize_counts(match_rows)
    summary.insert(0, "model", "one-step YOLO")
    per_class.insert(0, "model", "one-step YOLO")
    return summary, per_class, pd.DataFrame(match_rows)


def evaluate_two_step(
    matches_path: Path,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matches = pd.read_csv(matches_path)
    metadata_by_path = metadata.set_index("image_path")
    matches["lighting"] = matches["image_path"].map(metadata_by_path["lighting"]).fillna("unknown")
    matches["viewpoint"] = matches["image_path"].map(metadata_by_path["viewpoint"]).fillna("")
    matches["distance"] = matches["image_path"].map(metadata_by_path["distance"]).fillna("")
    matches["environment"] = matches["image_path"].map(metadata_by_path["environment"]).fillna("")
    matches["model"] = "two-step detector + classifier"
    records = matches[
        [
            "model",
            "image_path",
            "lighting",
            "viewpoint",
            "distance",
            "environment",
            "class_id",
            "class_name",
            "iou",
            "match_type",
        ]
    ].to_dict(orient="records")
    summary, per_class = summarize_counts(records)
    summary.insert(0, "model", "two-step detector + classifier")
    per_class.insert(0, "model", "two-step detector + classifier")
    return summary, per_class, pd.DataFrame(records)


def save_metric_plot(df: pd.DataFrame, output_path: Path, metric: str, condition_column: str = "lighting") -> None:
    pivot = df.pivot(index=condition_column, columns="model", values=metric).sort_index()
    ax = pivot.plot(kind="bar", figsize=(8, 4), ylim=(0, 1), rot=0)
    ax.set_title(f"{metric.upper()} by {condition_column}")
    ax.set_xlabel(condition_column.replace("_", " ").title())
    ax.set_ylabel(metric.upper())
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    predictions_path = args.one_step_predictions or find_one_step_predictions()

    raw_metadata = build_raw_metadata(args.raw_test)
    manual_metadata = load_manual_metadata(args.metadata)
    metadata = build_prepared_metadata(args.prepared, raw_metadata)
    metadata = apply_manual_metadata(metadata, manual_metadata)
    metadata = apply_condition_distribution(metadata, args.condition_distribution)
    gt_by_file = load_ground_truth(args.prepared, metadata)

    one_summary, one_per_class, one_matches = evaluate_one_step(
        predictions_path, metadata, gt_by_file, args.iou, args.one_step_conf
    )
    two_summary, two_per_class, two_matches = evaluate_two_step(args.two_step_matches, metadata)

    condition_summary = pd.concat([one_summary, two_summary], ignore_index=True)
    condition_per_class = pd.concat([one_per_class, two_per_class], ignore_index=True)
    condition_matches = pd.concat([one_matches, two_matches], ignore_index=True)

    metadata.to_csv(args.output / "test_condition_metadata.csv", index=False)
    condition_summary.to_csv(args.output / "condition_summary.csv", index=False)
    condition_per_class.to_csv(args.output / "condition_per_class.csv", index=False)
    condition_matches.to_csv(args.output / "condition_match_details.csv", index=False)
    raw_metadata.to_csv(args.output / "raw_test_metadata.csv", index=False)

    save_metric_plot(condition_summary, args.output / "condition_f1_by_lighting.png", "f1")
    save_metric_plot(condition_summary, args.output / "condition_precision_by_lighting.png", "precision")
    save_metric_plot(condition_summary, args.output / "condition_recall_by_lighting.png", "recall")

    optional_condition_columns = ["viewpoint", "distance", "environment"]
    for condition_column in optional_condition_columns:
        if condition_column not in condition_matches.columns:
            continue
        values = {
            str(value).strip()
            for value in condition_matches[condition_column].dropna().unique()
            if str(value).strip()
        }
        if not values:
            continue

        summaries = []
        per_class_tables = []
        for model_name, model_matches in condition_matches.groupby("model"):
            summary, per_class = summarize_counts(
                model_matches.to_dict(orient="records"),
                condition_column=condition_column,
            )
            if not summary.empty:
                summary.insert(0, "model", model_name)
                summaries.append(summary)
            if not per_class.empty:
                per_class.insert(0, "model", model_name)
                per_class_tables.append(per_class)

        if summaries:
            condition_df = pd.concat(summaries, ignore_index=True)
            condition_df.to_csv(args.output / f"condition_summary_by_{condition_column}.csv", index=False)
            save_metric_plot(
                condition_df,
                args.output / f"condition_f1_by_{condition_column}.png",
                "f1",
                condition_column=condition_column,
            )
        if per_class_tables:
            pd.concat(per_class_tables, ignore_index=True).to_csv(
                args.output / f"condition_per_class_by_{condition_column}.csv",
                index=False,
            )

    report_notes = {
        "one_step_predictions": str(predictions_path),
        "one_step_confidence_threshold": args.one_step_conf,
        "iou_threshold": args.iou,
        "prepared_test_images": int(len(metadata)),
        "raw_test_images": int(len(raw_metadata)),
        "prepared_images_without_raw_metadata": int((metadata["lighting"] == "unknown").sum()),
        "manual_metadata": str(args.metadata) if args.metadata else None,
        "condition_distribution": str(args.condition_distribution)
        if args.condition_distribution and args.condition_distribution.exists()
        else None,
    }
    (args.output / "report_asset_notes.json").write_text(
        json.dumps(report_notes, indent=2), encoding="utf-8"
    )

    print("Saved condition metadata and report assets to", args.output)
    print(condition_summary.to_string(index=False))


if __name__ == "__main__":
    main()
