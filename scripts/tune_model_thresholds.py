from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import cv2
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

from evaluate_conditions import (
    CLASS_NAMES,
    build_prepared_metadata,
    build_raw_metadata,
    find_one_step_predictions,
    iou,
    load_ground_truth,
    metric_row,
)
from low_light_enhancement import enhance_low_light


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_float_list(values: list[str] | None, defaults: list[float]) -> list[float]:
    if not values:
        return defaults
    return [float(value) for value in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune inference thresholds and crop padding.")
    parser.add_argument("--prepared", type=Path, default=PROJECT_ROOT / "prepared_dataset")
    parser.add_argument("--raw-test", type=Path, default=PROJECT_ROOT / "test_dataset")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "report_assets")
    parser.add_argument("--models", nargs="+", choices=["one-step", "two-step"], default=["one-step", "two-step"])
    parser.add_argument("--one-step-predictions", type=Path, default=None)
    parser.add_argument(
        "--one-step-thresholds",
        nargs="*",
        default=None,
        help="Confidence thresholds for saved one-step predictions.",
    )
    parser.add_argument("--one-step-weights", type=Path, default=PROJECT_ROOT / "runs/one_step/training/weights/best.pt")
    parser.add_argument("--person-weights", type=Path, default=PROJECT_ROOT / "yolov8n.pt")
    parser.add_argument("--classifier-weights", type=Path, default=PROJECT_ROOT / "runs/two_step/classifier_runs/best_classifier.pt")
    parser.add_argument("--person-confs", nargs="*", default=None)
    parser.add_argument("--crop-paddings", nargs="*", default=None)
    parser.add_argument("--enhancement", choices=["none", "gamma", "clahe", "gamma-clahe", "auto"], default="none")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.65)
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-tile-grid-size", type=int, default=8)
    parser.add_argument("--low-light-threshold", type=float, default=90.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def summarize_rows(rows: list[dict[str, object]], model: str, setting: str) -> dict[str, object]:
    tp = sum(1 for row in rows if row["match_type"] == "tp")
    fp = sum(1 for row in rows if row["match_type"] == "fp")
    fn = sum(1 for row in rows if row["match_type"] == "fn")
    summary = metric_row(setting, "all", tp, fp, fn, condition_column="setting")
    summary["model"] = model
    return summary


def summarize_per_class(rows: list[dict[str, object]], model: str, setting: str) -> list[dict[str, object]]:
    out = []
    df = pd.DataFrame(rows)
    if df.empty:
        return out
    for class_id, group in df.groupby("class_id"):
        tp = int((group["match_type"] == "tp").sum())
        fp = int((group["match_type"] == "fp").sum())
        fn = int((group["match_type"] == "fn").sum())
        row = metric_row(
            setting,
            CLASS_NAMES.get(int(class_id), str(class_id)),
            tp,
            fp,
            fn,
            int(class_id),
            condition_column="setting",
        )
        row["model"] = model
        out.append(row)
    return out


def match_predictions(
    image_key: str,
    gt_boxes: list[dict[str, float]],
    predictions: list[dict[str, float]],
    iou_threshold: float,
) -> list[dict[str, object]]:
    rows = []
    used = [False] * len(gt_boxes)
    for pred in sorted(predictions, key=lambda item: item.get("score", 0.0), reverse=True):
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
        rows.append(
            {
                "image": image_key,
                "class_id": pred["class_id"],
                "class_name": CLASS_NAMES[pred["class_id"]],
                "score": pred.get("score"),
                "iou": best_iou,
                "match_type": match_type,
            }
        )
    for idx, gt in enumerate(gt_boxes):
        if not used[idx]:
            rows.append(
                {
                    "image": image_key,
                    "class_id": gt["class_id"],
                    "class_name": CLASS_NAMES[gt["class_id"]],
                    "score": None,
                    "iou": None,
                    "match_type": "fn",
                }
            )
    return rows


def load_one_step_predictions(predictions_path: Path) -> dict[str, list[dict[str, float]]]:
    with predictions_path.open(encoding="utf-8") as json_file:
        raw_predictions = json.load(json_file)
    category_ids = {int(pred["category_id"]) for pred in raw_predictions}
    one_indexed_categories = category_ids and category_ids.issubset({1, 2, 3}) and 0 not in category_ids
    preds_by_file: dict[str, list[dict[str, float]]] = {}
    for pred in raw_predictions:
        class_id = int(pred["category_id"]) - 1 if one_indexed_categories else int(pred["category_id"])
        if class_id not in CLASS_NAMES:
            continue
        x, y, width, height = pred["bbox"]
        preds_by_file.setdefault(pred["file_name"], []).append(
            {
                "class_id": class_id,
                "score": float(pred.get("score", 0.0)),
                "x1": float(x),
                "y1": float(y),
                "x2": float(x + width),
                "y2": float(y + height),
            }
        )
    return preds_by_file


def tune_one_step(
    predictions_path: Path,
    gt_by_file: dict[str, list[dict[str, float]]],
    thresholds: list[float],
    iou_threshold: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    preds_by_file = load_one_step_predictions(predictions_path)
    summary_rows = []
    per_class_rows = []
    match_rows = []
    for threshold in thresholds:
        rows = []
        for file_name, gt_boxes in gt_by_file.items():
            predictions = [
                pred for pred in preds_by_file.get(file_name, [])
                if float(pred.get("score", 0.0)) >= threshold
            ]
            rows.extend(match_predictions(file_name, gt_boxes, predictions, iou_threshold))
        setting = f"conf={threshold:.2f}"
        summary_rows.append(summarize_rows(rows, "one-step YOLO", setting))
        per_class_rows.extend(summarize_per_class(rows, "one-step YOLO", setting))
        for row in rows:
            row.update({"model": "one-step YOLO", "setting": setting})
            match_rows.append(row)
    return summary_rows, per_class_rows, match_rows


def load_classifier(weights_path: Path, device: str) -> torch.nn.Module:
    classifier = models.resnet18(weights=None)
    classifier.fc = torch.nn.Linear(classifier.fc.in_features, len(CLASS_NAMES))
    classifier.load_state_dict(torch.load(weights_path, map_location=device))
    classifier.to(device)
    classifier.eval()
    return classifier


def classify_crop(
    classifier: torch.nn.Module,
    transform: transforms.Compose,
    crop_bgr: np.ndarray,
    device: str,
) -> tuple[int, float]:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(crop_rgb).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(classifier(tensor), dim=1)[0]
    class_id = int(probabilities.argmax().item())
    return class_id, float(probabilities[class_id].item())


def predict_two_step(
    image: np.ndarray,
    detector: YOLO,
    classifier: torch.nn.Module,
    transform: transforms.Compose,
    person_conf: float,
    crop_padding: float,
    imgsz: int,
    device: str,
) -> list[dict[str, float]]:
    result = detector.predict(image, classes=[0], conf=person_conf, imgsz=imgsz, verbose=False)[0]
    predictions = []
    if result.boxes is None:
        return predictions
    for idx in range(len(result.boxes)):
        x1, y1, x2, y2 = result.boxes.xyxy[idx].cpu().numpy().astype(int).tolist()
        score = float(result.boxes.conf[idx].item())
        box_width = x2 - x1
        box_height = y2 - y1
        pad_x = int(round(box_width * max(crop_padding, 0.0)))
        pad_y = int(round(box_height * max(crop_padding, 0.0)))
        px1, py1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        px2, py2 = min(image.shape[1], x2 + pad_x), min(image.shape[0], y2 + pad_y)
        crop = image[py1:py2, px1:px2]
        if crop.size == 0:
            continue
        class_id, class_conf = classify_crop(classifier, transform, crop, device)
        predictions.append(
            {
                "class_id": class_id,
                "score": score * class_conf,
                "detector_score": score,
                "classifier_score": class_conf,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }
        )
    return predictions


def tune_two_step(
    metadata: pd.DataFrame,
    gt_by_file: dict[str, list[dict[str, float]]],
    args: argparse.Namespace,
    person_confs: list[float],
    crop_paddings: list[float],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    detector = YOLO(str(args.person_weights))
    classifier = load_classifier(args.classifier_weights, args.device)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    summary_rows = []
    per_class_rows = []
    match_rows = []
    for person_conf in person_confs:
        for crop_padding in crop_paddings:
            rows = []
            for record in metadata.itertuples(index=False):
                image = cv2.imread(record.image_path)
                if image is None:
                    continue
                enhanced, _ = enhance_low_light(
                    image,
                    mode=args.enhancement,
                    gamma=args.gamma,
                    clahe_clip_limit=args.clahe_clip_limit,
                    clahe_tile_grid_size=args.clahe_tile_grid_size,
                    low_light_threshold=args.low_light_threshold,
                )
                predictions = predict_two_step(
                    enhanced,
                    detector,
                    classifier,
                    transform,
                    person_conf,
                    crop_padding,
                    args.imgsz,
                    args.device,
                )
                rows.extend(match_predictions(record.file_name, gt_by_file.get(record.file_name, []), predictions, args.iou))
            setting = f"person_conf={person_conf:.2f},padding={crop_padding:.2f},enh={args.enhancement}"
            summary_rows.append(summarize_rows(rows, "two-step detector + classifier", setting))
            per_class_rows.extend(summarize_per_class(rows, "two-step detector + classifier", setting))
            for row in rows:
                row.update({"model": "two-step detector + classifier", "setting": setting})
                match_rows.append(row)
    return summary_rows, per_class_rows, match_rows


def save_plot(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    ranked = summary.sort_values("f1", ascending=False).head(12).copy()
    ranked["label"] = ranked["model"] + "\n" + ranked["setting"]
    ax = ranked.plot(kind="bar", x="label", y="f1", figsize=(11, 5), legend=False, ylim=(0, 1), rot=45)
    ax.set_title("Top Tuned Inference Settings by F1")
    ax.set_xlabel("Setting")
    ax.set_ylabel("F1")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    one_step_thresholds = parse_float_list(args.one_step_thresholds, [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60])
    person_confs = parse_float_list(args.person_confs, [0.15, 0.20, 0.25, 0.30, 0.35])
    crop_paddings = parse_float_list(args.crop_paddings, [0.00, 0.05, 0.10, 0.15, 0.20])
    raw_metadata = build_raw_metadata(args.raw_test)
    metadata = build_prepared_metadata(args.prepared, raw_metadata)
    gt_by_file = load_ground_truth(args.prepared, metadata)

    summary_rows = []
    per_class_rows = []
    match_rows = []
    if "one-step" in args.models:
        predictions_path = args.one_step_predictions or find_one_step_predictions()
        one_summary, one_per_class, one_matches = tune_one_step(predictions_path, gt_by_file, one_step_thresholds, args.iou)
        summary_rows.extend(one_summary)
        per_class_rows.extend(one_per_class)
        match_rows.extend(one_matches)
    if "two-step" in args.models:
        two_summary, two_per_class, two_matches = tune_two_step(metadata, gt_by_file, args, person_confs, crop_paddings)
        summary_rows.extend(two_summary)
        per_class_rows.extend(two_per_class)
        match_rows.extend(two_matches)

    summary = pd.DataFrame(summary_rows)
    per_class = pd.DataFrame(per_class_rows)
    matches = pd.DataFrame(match_rows)
    preferred = ["model", "setting", "tp", "fp", "fn", "precision", "recall", "f1"]
    class_preferred = ["model", "setting", "class_id", "class_name", "tp", "fp", "fn", "precision", "recall", "f1"]
    summary = summary[[column for column in preferred if column in summary.columns]].sort_values(["model", "f1"], ascending=[True, False])
    per_class = per_class[[column for column in class_preferred if column in per_class.columns]]
    summary.to_csv(args.output / "tuned_inference_summary.csv", index=False)
    per_class.to_csv(args.output / "tuned_inference_per_class.csv", index=False)
    matches.to_csv(args.output / "tuned_inference_match_details.csv", index=False)
    save_plot(summary, args.output / "tuned_inference_top_f1.png")
    print("Saved tuned inference assets to", args.output)
    print(summary.groupby("model").head(5).to_string(index=False))


if __name__ == "__main__":
    main()
