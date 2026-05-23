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
    apply_manual_metadata,
    build_prepared_metadata,
    build_raw_metadata,
    iou,
    load_ground_truth,
    load_manual_metadata,
    metric_row,
)
from low_light_enhancement import ENHANCEMENT_MODES, enhance_low_light, mean_luminance


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate low-light enhancement extension.")
    parser.add_argument("--prepared", type=Path, default=PROJECT_ROOT / "prepared_dataset")
    parser.add_argument("--raw-test", type=Path, default=PROJECT_ROOT / "test_dataset")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional metadata CSV, matching scripts/evaluate_conditions.py.",
    )
    parser.add_argument(
        "--one-step-weights",
        type=Path,
        default=PROJECT_ROOT / "runs" / "baseline_runs" / "one_step_baseline" / "weights" / "best.pt",
    )
    parser.add_argument("--person-weights", type=Path, default=PROJECT_ROOT / "yolov8n.pt")
    parser.add_argument(
        "--classifier-weights",
        type=Path,
        default=PROJECT_ROOT / "runs" / "two_step" / "classifier_runs" / "best_classifier.pt",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "report_assets")
    parser.add_argument("--modes", nargs="+", choices=ENHANCEMENT_MODES, default=list(ENHANCEMENT_MODES))
    parser.add_argument("--models", nargs="+", choices=["one-step", "two-step"], default=["one-step", "two-step"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.65)
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-tile-grid-size", type=int, default=8)
    parser.add_argument("--low-light-threshold", type=float, default=90.0)
    parser.add_argument("--crop-padding", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--example-count", type=int, default=4)
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")


def load_classifier(weights_path: Path, device: str) -> torch.nn.Module:
    classifier = models.resnet18(weights=None)
    classifier.fc = torch.nn.Linear(classifier.fc.in_features, len(CLASS_NAMES))
    state_dict = torch.load(weights_path, map_location=device)
    classifier.load_state_dict(state_dict)
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


def match_predictions(
    model_name: str,
    mode: str,
    file_name: str,
    image_path: str,
    gt_boxes: list[dict[str, float]],
    predictions: list[dict[str, float]],
    iou_threshold: float,
    luminance: float,
    enhancement_applied: bool,
) -> list[dict[str, object]]:
    rows = []
    used = [False] * len(gt_boxes)
    for pred in sorted(predictions, key=lambda item: item["score"], reverse=True):
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
                "model": model_name,
                "enhancement": mode,
                "file_name": file_name,
                "image_path": image_path,
                "class_id": pred["class_id"],
                "class_name": CLASS_NAMES[pred["class_id"]],
                "score": pred["score"],
                "iou": best_iou,
                "match_type": match_type,
                "luminance": luminance,
                "enhancement_applied": enhancement_applied,
            }
        )

    for idx, gt in enumerate(gt_boxes):
        if not used[idx]:
            rows.append(
                {
                    "model": model_name,
                    "enhancement": mode,
                    "file_name": file_name,
                    "image_path": image_path,
                    "class_id": gt["class_id"],
                    "class_name": CLASS_NAMES[gt["class_id"]],
                    "score": None,
                    "iou": None,
                    "match_type": "fn",
                    "luminance": luminance,
                    "enhancement_applied": enhancement_applied,
                }
            )
    return rows


def summarize_matches(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    per_class_rows = []
    for (model_name, enhancement), group in matches.groupby(["model", "enhancement"], dropna=False):
        tp = int((group["match_type"] == "tp").sum())
        fp = int((group["match_type"] == "fp").sum())
        fn = int((group["match_type"] == "fn").sum())
        row = metric_row(str(enhancement), "all", tp, fp, fn, condition_column="enhancement")
        row["model"] = model_name
        summary_rows.append(row)
        for class_id, class_group in group.groupby("class_id", dropna=False):
            tp = int((class_group["match_type"] == "tp").sum())
            fp = int((class_group["match_type"] == "fp").sum())
            fn = int((class_group["match_type"] == "fn").sum())
            class_row = metric_row(
                str(enhancement),
                CLASS_NAMES.get(int(class_id), str(class_id)),
                tp,
                fp,
                fn,
                int(class_id),
                condition_column="enhancement",
            )
            class_row["model"] = model_name
            per_class_rows.append(class_row)
    summary = pd.DataFrame(summary_rows)
    per_class = pd.DataFrame(per_class_rows)
    preferred = ["model", "enhancement", "class_id", "class_name", "tp", "fp", "fn", "precision", "recall", "f1"]
    summary = summary[[column for column in preferred if column in summary.columns]]
    per_class = per_class[[column for column in preferred if column in per_class.columns]]
    return summary, per_class


def predict_one_step(model: YOLO, image: np.ndarray, conf: float, imgsz: int) -> list[dict[str, float]]:
    result = model.predict(image, conf=conf, imgsz=imgsz, verbose=False)[0]
    predictions = []
    if result.boxes is None:
        return predictions
    for idx in range(len(result.boxes)):
        class_id = int(result.boxes.cls[idx].item())
        if class_id not in CLASS_NAMES:
            continue
        confidence = float(result.boxes.conf[idx].item())
        x1, y1, x2, y2 = result.boxes.xyxy[idx].cpu().numpy().astype(float).tolist()
        predictions.append(
            {"class_id": class_id, "score": confidence, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
        )
    return predictions


def predict_two_step(
    detector: YOLO,
    classifier: torch.nn.Module,
    transform: transforms.Compose,
    image: np.ndarray,
    conf: float,
    imgsz: int,
    device: str,
    crop_padding: float,
) -> list[dict[str, float]]:
    result = detector.predict(image, classes=[0], conf=conf, imgsz=imgsz, verbose=False)[0]
    predictions = []
    if result.boxes is None:
        return predictions
    for idx in range(len(result.boxes)):
        x1, y1, x2, y2 = result.boxes.xyxy[idx].cpu().numpy().astype(int).tolist()
        box_width = x2 - x1
        box_height = y2 - y1
        pad_x = int(round(box_width * max(crop_padding, 0.0)))
        pad_y = int(round(box_height * max(crop_padding, 0.0)))
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(image.shape[1], x2 + pad_x), min(image.shape[0], y2 + pad_y)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        class_id, confidence = classify_crop(classifier, transform, crop, device)
        predictions.append(
            {
                "class_id": class_id,
                "score": confidence,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }
        )
    return predictions


def save_examples(
    metadata: pd.DataFrame,
    modes: list[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in metadata.head(args.example_count).itertuples(index=False):
        image = cv2.imread(row.image_path)
        if image is None:
            continue
        cv2.imwrite(str(output_dir / f"{row.stem}_original.jpg"), image)
        for mode in modes:
            if mode == "none":
                continue
            enhanced, _ = enhance_low_light(
                image,
                mode=mode,
                gamma=args.gamma,
                clahe_clip_limit=args.clahe_clip_limit,
                clahe_tile_grid_size=args.clahe_tile_grid_size,
                low_light_threshold=args.low_light_threshold,
            )
            cv2.imwrite(str(output_dir / f"{row.stem}_{mode}.jpg"), enhanced)


def save_plot(summary: pd.DataFrame, output_path: Path) -> None:
    pivot = summary.pivot(index="enhancement", columns="model", values="f1")
    order = [mode for mode in ENHANCEMENT_MODES if mode in pivot.index]
    pivot = pivot.loc[order]
    ax = pivot.plot(kind="bar", figsize=(8, 4), ylim=(0, 1), rot=20)
    ax.set_title("Low-Light Extension F1 by Enhancement")
    ax.set_xlabel("Enhancement Mode")
    ax.set_ylabel("F1")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    require_file(args.one_step_weights, "one-step weights") if "one-step" in args.models else None
    require_file(args.person_weights, "person detector weights") if "two-step" in args.models else None
    require_file(args.classifier_weights, "classifier weights") if "two-step" in args.models else None

    raw_metadata = build_raw_metadata(args.raw_test)
    manual_metadata = load_manual_metadata(args.metadata)
    metadata = build_prepared_metadata(args.prepared, raw_metadata)
    metadata = apply_manual_metadata(metadata, manual_metadata)
    metadata = metadata[metadata["lighting"].astype(str).str.lower() == "low light"].copy()
    if metadata.empty:
        raise SystemExit("No Low Light images found in prepared test metadata.")
    gt_by_file = load_ground_truth(args.prepared, metadata)

    one_step_model = YOLO(str(args.one_step_weights)) if "one-step" in args.models else None
    detector = YOLO(str(args.person_weights)) if "two-step" in args.models else None
    classifier = load_classifier(args.classifier_weights, args.device) if "two-step" in args.models else None
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    match_rows = []
    for mode in args.modes:
        for row in metadata.itertuples(index=False):
            image = cv2.imread(row.image_path)
            if image is None:
                continue
            luminance = mean_luminance(image)
            enhanced, applied = enhance_low_light(
                image,
                mode=mode,
                gamma=args.gamma,
                clahe_clip_limit=args.clahe_clip_limit,
                clahe_tile_grid_size=args.clahe_tile_grid_size,
                low_light_threshold=args.low_light_threshold,
            )
            gt_boxes = gt_by_file.get(row.file_name, [])
            if one_step_model is not None:
                predictions = predict_one_step(one_step_model, enhanced, args.conf, args.imgsz)
                match_rows.extend(
                    match_predictions(
                        "one-step YOLO",
                        mode,
                        row.file_name,
                        row.image_path,
                        gt_boxes,
                        predictions,
                        args.iou,
                        luminance,
                        applied,
                    )
                )
            if detector is not None and classifier is not None:
                predictions = predict_two_step(
                    detector,
                    classifier,
                    transform,
                    enhanced,
                    args.conf,
                    args.imgsz,
                    args.device,
                    args.crop_padding,
                )
                match_rows.extend(
                    match_predictions(
                        "two-step detector + classifier",
                        mode,
                        row.file_name,
                        row.image_path,
                        gt_boxes,
                        predictions,
                        args.iou,
                        luminance,
                        applied,
                    )
                )

    matches = pd.DataFrame(match_rows)
    summary, per_class = summarize_matches(matches)
    matches.to_csv(args.output / "low_light_extension_match_details.csv", index=False)
    summary.to_csv(args.output / "low_light_extension_summary.csv", index=False)
    per_class.to_csv(args.output / "low_light_extension_per_class.csv", index=False)
    save_plot(summary, args.output / "low_light_extension_f1.png")
    save_examples(
        metadata,
        list(args.modes),
        args.output / "low_light_extension_examples",
        args,
    )

    notes = {
        "modes": list(args.modes),
        "models": list(args.models),
        "low_light_images": int(len(metadata)),
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "gamma": args.gamma,
        "clahe_clip_limit": args.clahe_clip_limit,
        "clahe_tile_grid_size": args.clahe_tile_grid_size,
        "low_light_threshold": args.low_light_threshold,
        "crop_padding": args.crop_padding,
    }
    (args.output / "low_light_extension_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")

    print("Saved low-light extension assets to", args.output)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
