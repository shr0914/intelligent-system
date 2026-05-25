from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import cv2
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
import matplotlib.pyplot as plt
import pandas as pd
import torch

from evaluate_conditions import (
    CLASS_NAMES,
    apply_manual_metadata,
    build_prepared_metadata,
    build_raw_metadata,
    iou,
    load_ground_truth,
    load_manual_metadata,
    metric_row,
    summarize_counts,
)
from low_light_enhancement import ENHANCEMENT_MODES, enhance_low_light, mean_luminance
from pose_fall_extension import (
    PoseRuleConfig,
    draw_pose_prediction,
    load_pose_model,
    predict_pose,
    serializable_prediction,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pose-estimation fall-detection extension.")
    parser.add_argument("--prepared", type=Path, default=PROJECT_ROOT / "prepared_dataset")
    parser.add_argument("--raw-test", type=Path, default=PROJECT_ROOT / "test_dataset")
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--pose-weights", type=Path, default=PROJECT_ROOT / "yolov8n-pose.pt")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "pose_extension")
    parser.add_argument("--report-output", type=Path, default=PROJECT_ROOT / "runs" / "report_assets")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--fall-angle-threshold", type=float, default=55.0)
    parser.add_argument("--fall-ratio-threshold", type=float, default=0.90)
    parser.add_argument("--fall-vertical-spread-threshold", type=float, default=0.80)
    parser.add_argument("--sit-ratio-threshold", type=float, default=0.45)
    parser.add_argument("--sit-knee-hip-threshold", type=float, default=0.15)
    parser.add_argument("--min-keypoint-conf", type=float, default=0.25)
    parser.add_argument("--min-visible-keypoints", type=int, default=5)
    parser.add_argument("--enhancements", nargs="+", choices=ENHANCEMENT_MODES, default=["none"])
    parser.add_argument("--low-light-only", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.65)
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-tile-grid-size", type=int, default=8)
    parser.add_argument("--low-light-threshold", type=float, default=90.0)
    parser.add_argument("--example-count", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> PoseRuleConfig:
    return PoseRuleConfig(
        fall_angle_threshold=args.fall_angle_threshold,
        fall_ratio_threshold=args.fall_ratio_threshold,
        fall_vertical_spread_threshold=args.fall_vertical_spread_threshold,
        sit_ratio_threshold=args.sit_ratio_threshold,
        sit_knee_hip_threshold=args.sit_knee_hip_threshold,
        min_keypoint_conf=args.min_keypoint_conf,
        min_visible_keypoints=args.min_visible_keypoints,
    )


def match_predictions(
    enhancement: str,
    file_name: str,
    image_path: str,
    gt_boxes: list[dict[str, float]],
    predictions: list[dict[str, object]],
    metadata_row: pd.Series | dict[str, object],
    iou_threshold: float,
) -> list[dict[str, object]]:
    rows = []
    used = [False] * len(gt_boxes)
    lighting = metadata_row.get("lighting", "unknown") if isinstance(metadata_row, pd.Series) else "unknown"
    viewpoint = metadata_row.get("viewpoint", "") if isinstance(metadata_row, pd.Series) else ""
    distance = metadata_row.get("distance", "") if isinstance(metadata_row, pd.Series) else ""
    environment = metadata_row.get("environment", "") if isinstance(metadata_row, pd.Series) else ""

    for pred in sorted(predictions, key=lambda item: float(item["score"]), reverse=True):
        best_iou = 0.0
        best_idx = None
        for idx, gt in enumerate(gt_boxes):
            if used[idx] or int(gt["class_id"]) != int(pred["class_id"]):
                continue
            score = iou(pred, gt)
            if score > best_iou:
                best_iou = score
                best_idx = idx
        match_type = "fp"
        if best_idx is not None and best_iou >= iou_threshold:
            used[best_idx] = True
            match_type = "tp"
        rows.append(
            {
                "model": "pose-rule extension",
                "enhancement": enhancement,
                "file_name": file_name,
                "image_path": image_path,
                "lighting": lighting,
                "viewpoint": viewpoint,
                "distance": distance,
                "environment": environment,
                "class_id": int(pred["class_id"]),
                "class_name": CLASS_NAMES[int(pred["class_id"])],
                "score": float(pred["score"]),
                "detector_score": float(pred["detector_score"]),
                "iou": best_iou,
                "match_type": match_type,
                "rule_reason": pred["rule_reason"],
                "box_ratio": pred["box_ratio"],
                "torso_angle_from_horizontal": pred["torso_angle_from_horizontal"],
                "vertical_spread": pred["vertical_spread"],
                "hip_knee_gap": pred["hip_knee_gap"],
                "visible_keypoints": pred["visible_keypoints"],
                "avg_keypoint_conf": pred["avg_keypoint_conf"],
            }
        )

    for idx, gt in enumerate(gt_boxes):
        if not used[idx]:
            rows.append(
                {
                    "model": "pose-rule extension",
                    "enhancement": enhancement,
                    "file_name": file_name,
                    "image_path": image_path,
                    "lighting": lighting,
                    "viewpoint": viewpoint,
                    "distance": distance,
                    "environment": environment,
                    "class_id": int(gt["class_id"]),
                    "class_name": CLASS_NAMES[int(gt["class_id"])],
                    "score": None,
                    "detector_score": None,
                    "iou": None,
                    "match_type": "fn",
                    "rule_reason": "missed_ground_truth",
                    "box_ratio": None,
                    "torso_angle_from_horizontal": None,
                    "vertical_spread": None,
                    "hip_knee_gap": None,
                    "visible_keypoints": None,
                    "avg_keypoint_conf": None,
                }
            )
    return rows


def summarize_overall(matches: pd.DataFrame, group_column: str = "split") -> tuple[pd.DataFrame, pd.DataFrame]:
    if group_column in matches.columns:
        summary_rows = []
        per_class_rows = []
        for group_value, group in matches.groupby(group_column, dropna=False):
            tp = int((group["match_type"] == "tp").sum())
            fp = int((group["match_type"] == "fp").sum())
            fn = int((group["match_type"] == "fn").sum())
            summary_rows.append(metric_row(str(group_value), "all", tp, fp, fn, condition_column=group_column))
            for class_id, class_group in group.groupby("class_id", dropna=False):
                tp = int((class_group["match_type"] == "tp").sum())
                fp = int((class_group["match_type"] == "fp").sum())
                fn = int((class_group["match_type"] == "fn").sum())
                per_class_rows.append(
                    metric_row(
                        str(group_value),
                        CLASS_NAMES.get(int(class_id), str(class_id)),
                        tp,
                        fp,
                        fn,
                        int(class_id),
                        condition_column=group_column,
                    )
                )
        summary = pd.DataFrame(summary_rows)
        per_class = pd.DataFrame(per_class_rows)
        summary.insert(0, "model", "pose-rule extension")
        per_class.insert(0, "model", "pose-rule extension")
        return summary, per_class

    tp = int((matches["match_type"] == "tp").sum())
    fp = int((matches["match_type"] == "fp").sum())
    fn = int((matches["match_type"] == "fn").sum())
    summary = pd.DataFrame([metric_row("all", "all", tp, fp, fn, condition_column="split")])
    summary.insert(0, "model", "pose-rule extension")

    rows = []
    for class_id, group in matches.groupby("class_id", dropna=False):
        tp = int((group["match_type"] == "tp").sum())
        fp = int((group["match_type"] == "fp").sum())
        fn = int((group["match_type"] == "fn").sum())
        rows.append(
            metric_row(
                "all",
                CLASS_NAMES.get(int(class_id), str(class_id)),
                tp,
                fp,
                fn,
                int(class_id),
                condition_column="split",
            )
        )
    per_class = pd.DataFrame(rows)
    per_class.insert(0, "model", "pose-rule extension")
    return summary, per_class


def save_condition_tables(matches: pd.DataFrame, output: Path, report_output: Path) -> None:
    records = matches.to_dict(orient="records")
    condition_summary, condition_per_class = summarize_counts(records)
    condition_summary.insert(0, "model", "pose-rule extension")
    condition_per_class.insert(0, "model", "pose-rule extension")
    condition_summary.to_csv(output / "pose_condition_summary.csv", index=False)
    condition_per_class.to_csv(output / "pose_condition_per_class.csv", index=False)
    condition_summary.to_csv(report_output / "pose_condition_summary.csv", index=False)
    condition_per_class.to_csv(report_output / "pose_condition_per_class.csv", index=False)

    if not condition_summary.empty:
        ax = condition_summary.set_index("lighting")["f1"].sort_index().plot(
            kind="bar",
            figsize=(7, 4),
            ylim=(0, 1),
            rot=0,
            color="#5874d8",
        )
        ax.set_title("Pose Extension F1 by Lighting")
        ax.set_xlabel("Lighting")
        ax.set_ylabel("F1")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output / "pose_f1_by_lighting.png", dpi=160)
        plt.savefig(report_output / "pose_f1_by_lighting.png", dpi=160)
        plt.close()


def save_enhancement_plot(summary: pd.DataFrame, output: Path, report_output: Path) -> None:
    if summary.empty or "enhancement" not in summary.columns:
        return
    order = [mode for mode in ENHANCEMENT_MODES if mode in set(summary["enhancement"])]
    plot_data = summary.set_index("enhancement").loc[order]
    ax = plot_data["f1"].plot(kind="bar", figsize=(8, 4), ylim=(0, 1), rot=20, color="#5874d8")
    ax.set_title("Low-Light Pose Pipeline F1 by Enhancement")
    ax.set_xlabel("Enhancement Mode")
    ax.set_ylabel("F1")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / "pose_low_light_enhancement_f1.png", dpi=160)
    plt.savefig(report_output / "pose_low_light_enhancement_f1.png", dpi=160)
    plt.close()


def save_examples(
    predictions_by_file: dict[str, list[dict[str, object]]],
    metadata: pd.DataFrame,
    output_dir: Path,
    example_count: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for row in metadata.itertuples(index=False):
        predictions = predictions_by_file.get(row.file_name, [])
        if not predictions:
            continue
        image = cv2.imread(row.image_path)
        if image is None:
            continue
        for prediction in predictions:
            draw_pose_prediction(image, prediction)
        cv2.imwrite(str(output_dir / f"{row.stem}_pose.jpg"), image)
        written += 1
        if written >= example_count:
            break


def save_comparison_update(report_output: Path, pose_summary: pd.DataFrame) -> None:
    comparison_path = report_output.parent / "comparison" / "comparison_summary.csv"
    if not comparison_path.exists() or pose_summary.empty:
        return
    comparison = pd.read_csv(comparison_path)
    pose_row = pose_summary.iloc[0].to_dict()
    extension_row = {
        "model": "pose-rule extension",
        "precision": pose_row.get("precision"),
        "recall": pose_row.get("recall"),
        "f1": pose_row.get("f1"),
        "mAP50": None,
        "mAP50-95": None,
        "crop_test_accuracy": None,
        "notes": "YOLO pose keypoint geometry with rule-based posture classification",
    }
    comparison = comparison[comparison["model"] != "pose-rule extension"]
    comparison = pd.concat([comparison, pd.DataFrame([extension_row])], ignore_index=True)
    comparison.to_csv(report_output / "comparison_with_pose_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.report_output.mkdir(parents=True, exist_ok=True)

    raw_metadata = build_raw_metadata(args.raw_test)
    manual_metadata = load_manual_metadata(args.metadata)
    metadata = build_prepared_metadata(args.prepared, raw_metadata)
    metadata = apply_manual_metadata(metadata, manual_metadata)
    if args.low_light_only:
        metadata = metadata[metadata["lighting"].astype(str).str.lower() == "low light"].copy()
        if metadata.empty:
            raise SystemExit("No Low Light images found in prepared test metadata.")
    gt_by_file = load_ground_truth(args.prepared, metadata)
    metadata_by_file = metadata.set_index("file_name")

    model = load_pose_model(args.pose_weights)
    config = config_from_args(args)

    match_rows = []
    prediction_rows = []
    predictions_by_file: dict[str, list[dict[str, object]]] = {}
    for enhancement in args.enhancements:
        for row in metadata.itertuples(index=False):
            image = cv2.imread(row.image_path)
            if image is None:
                continue
            luminance = mean_luminance(image)
            enhanced, applied = enhance_low_light(
                image,
                mode=enhancement,
                gamma=args.gamma,
                clahe_clip_limit=args.clahe_clip_limit,
                clahe_tile_grid_size=args.clahe_tile_grid_size,
                low_light_threshold=args.low_light_threshold,
            )
            predictions = predict_pose(model, enhanced, args.conf, args.imgsz, config)
            if enhancement == args.enhancements[0]:
                predictions_by_file[row.file_name] = predictions
            prediction_rows.extend(
                {
                    "enhancement": enhancement,
                    "enhancement_applied": applied,
                    "luminance": luminance,
                    "file_name": row.file_name,
                    "image_path": row.image_path,
                    **serializable_prediction(prediction),
                }
                for prediction in predictions
            )
            match_rows.extend(
                match_predictions(
                    enhancement,
                    row.file_name,
                    row.image_path,
                    gt_by_file.get(row.file_name, []),
                    predictions,
                    metadata_by_file.loc[row.file_name] if row.file_name in metadata_by_file.index else {},
                    args.iou,
                )
            )

    matches = pd.DataFrame(match_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary, per_class = summarize_overall(matches, "enhancement" if len(args.enhancements) > 1 else "split")

    matches.to_csv(args.output / "pose_match_details.csv", index=False)
    predictions.to_csv(args.output / "pose_predictions.csv", index=False)
    summary.to_csv(args.output / "pose_metrics_summary.csv", index=False)
    per_class.to_csv(args.output / "pose_per_class_metrics.csv", index=False)
    matches.to_csv(args.report_output / "pose_match_details.csv", index=False)
    summary.to_csv(args.report_output / "pose_metrics_summary.csv", index=False)
    per_class.to_csv(args.report_output / "pose_per_class_metrics.csv", index=False)
    save_condition_tables(matches, args.output, args.report_output)
    save_enhancement_plot(summary, args.output, args.report_output)
    if args.low_light_only and len(args.enhancements) > 1:
        summary.to_csv(args.output / "pose_low_light_enhancement_summary.csv", index=False)
        per_class.to_csv(args.output / "pose_low_light_enhancement_per_class.csv", index=False)
        matches.to_csv(args.output / "pose_low_light_enhancement_match_details.csv", index=False)
        summary.to_csv(args.report_output / "pose_low_light_enhancement_summary.csv", index=False)
        per_class.to_csv(args.report_output / "pose_low_light_enhancement_per_class.csv", index=False)
        matches.to_csv(args.report_output / "pose_low_light_enhancement_match_details.csv", index=False)
    save_examples(predictions_by_file, metadata, args.output / "examples", args.example_count)
    if len(args.enhancements) == 1:
        save_comparison_update(args.report_output, summary)

    notes = {
        "pose_weights": str(args.pose_weights),
        "prepared_test_images": int(len(metadata)),
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "imgsz": args.imgsz,
        "device": args.device,
        "enhancements": list(args.enhancements),
        "low_light_only": args.low_light_only,
        "gamma": args.gamma,
        "clahe_clip_limit": args.clahe_clip_limit,
        "clahe_tile_grid_size": args.clahe_tile_grid_size,
        "low_light_threshold": args.low_light_threshold,
        "rule_config": config.__dict__,
        "method": "YOLO pose estimates keypoints, then transparent posture rules classify fall/walk/sit.",
    }
    (args.output / "pose_extension_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    (args.report_output / "pose_extension_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")

    print("Saved pose extension assets to", args.output)
    print(summary.to_string(index=False))
    print()
    print(per_class.to_string(index=False))


if __name__ == "__main__":
    main()
