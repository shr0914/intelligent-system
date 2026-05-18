from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = {
    0: "fall detected",
    1: "walk",
    2: "sit",
}
CLASS_COLORS = {
    0: (58, 78, 239),
    1: (92, 184, 92),
    2: (235, 176, 72),
}
CLASS_DISPLAY_NAMES = {
    0: "Fall",
    1: "Walk",
    2: "Sit",
}
PANEL_WIDTH = 400
PADDING = 22
UI_BG = (18, 20, 24)
PANEL_BG = (24, 27, 33)
CARD_BG = (35, 39, 47)
CARD_BORDER = (58, 63, 74)
TEXT = (244, 246, 248)
TEXT_MUTED = (158, 166, 178)
TEXT_DIM = (116, 124, 138)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live fall-detection GUI demo.")
    parser.add_argument(
        "--mode",
        choices=["one-step", "two-step"],
        default="one-step",
        help="Inference pipeline to run.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index, video path, image path, or folder path. Default: 0.",
    )
    parser.add_argument(
        "--one-step-weights",
        type=Path,
        default=PROJECT_ROOT / "runs" / "baseline_runs" / "one_step_baseline" / "weights" / "best.pt",
    )
    parser.add_argument(
        "--person-weights",
        type=Path,
        default=PROJECT_ROOT / "yolov8n.pt",
    )
    parser.add_argument(
        "--classifier-weights",
        type=Path,
        default=PROJECT_ROOT / "runs" / "two_step" / "classifier_runs" / "best_classifier.pt",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--camera-width",
        type=int,
        default=1280,
        help="Requested webcam capture width.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=720,
        help="Requested webcam capture height.",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=1280,
        help="Displayed camera-feed width before adding the side panel.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Process frames without opening an OpenCV window. Useful on headless machines.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for an annotated output video.",
    )
    return parser.parse_args()


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


def draw_detection(
    frame: np.ndarray,
    class_id: int,
    confidence: float,
    box: tuple[int, int, int, int],
) -> None:
    x1, y1, x2, y2 = box
    color = CLASS_COLORS.get(class_id, (255, 255, 255))
    label = f"{CLASS_DISPLAY_NAMES.get(class_id, class_id)} {confidence:.2f}"
    thickness = max(2, round(frame.shape[1] / 700))
    font_scale = max(0.6, min(0.9, frame.shape[1] / 1500))
    font_thickness = max(2, round(font_scale * 2))
    text_size, baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        font_thickness,
    )
    label_width = text_size[0] + 18
    label_height = text_size[1] + baseline + 12
    label_x1 = max(0, min(x1, frame.shape[1] - label_width))
    label_y1 = y1 - label_height if y1 >= label_height + 4 else min(y2 + 4, frame.shape[0] - label_height)
    label_y2 = label_y1 + label_height

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.rectangle(frame, (label_x1, label_y1), (label_x1 + label_width, label_y2), (22, 24, 29), -1)
    cv2.rectangle(frame, (label_x1, label_y1), (label_x1 + 5, label_y2), color, -1)
    cv2.putText(
        frame,
        label,
        (label_x1 + 12, label_y2 - baseline - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        font_thickness,
        cv2.LINE_AA,
    )


def draw_card(panel: np.ndarray, x: int, y: int, width: int, height: int, fill: tuple[int, int, int] = CARD_BG) -> None:
    cv2.rectangle(panel, (x, y), (x + width, y + height), fill, -1)
    cv2.rectangle(panel, (x, y), (x + width, y + height), CARD_BORDER, 1)


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_panel(
    canvas: np.ndarray,
    panel_x: int,
    fall_detected: bool,
    mode: str,
    detections: list[dict[str, object]],
    fps: float,
    conf: float,
) -> None:
    panel = canvas[:, panel_x:]
    panel[:] = PANEL_BG
    cv2.line(canvas, (panel_x, 0), (panel_x, canvas.shape[0]), (54, 59, 70), 1)

    put_text(panel, "Fall Detection", (PADDING, 38), 0.82, TEXT, 2)
    put_text(panel, "Live vision prototype", (PADDING, 66), 0.48, TEXT_MUTED, 1)

    status_color = CLASS_COLORS[0] if fall_detected else (72, 178, 112)
    status_fill = (37, 43, 61) if fall_detected else (32, 48, 39)
    status_text = "FALL ALERT" if fall_detected else "NORMAL"
    draw_card(panel, PADDING, 94, panel.shape[1] - PADDING * 2, 116, status_fill)
    cv2.circle(panel, (PADDING + 28, 128), 10, status_color, -1)
    put_text(panel, status_text, (PADDING + 50, 137), 0.86, TEXT, 2)
    put_text(panel, f"{mode.replace('-', ' ').title()} pipeline", (PADDING + 22, 176), 0.52, TEXT_MUTED, 1)

    card_y = 232
    card_w = (panel.shape[1] - PADDING * 2 - 12) // 2
    metrics = [
        ("Objects", str(len(detections))),
        ("FPS", f"{fps:.1f}"),
    ]
    for index, (label, value) in enumerate(metrics):
        x = PADDING + index * (card_w + 12)
        draw_card(panel, x, card_y, card_w, 86)
        put_text(panel, label, (x + 16, card_y + 29), 0.48, TEXT_MUTED, 1)
        put_text(panel, value, (x + 16, card_y + 66), 0.84, TEXT, 2)

    draw_card(panel, PADDING, card_y + 106, panel.shape[1] - PADDING * 2, 64)
    put_text(panel, "Confidence threshold", (PADDING + 16, card_y + 132), 0.48, TEXT_MUTED, 1)
    put_text(panel, f"{conf:.2f}", (panel.shape[1] - PADDING - 76, card_y + 138), 0.76, TEXT, 2)

    counts = {class_id: 0 for class_id in CLASS_NAMES}
    for detection in detections:
        counts[int(detection["class_id"])] += 1

    y = card_y + 204
    put_text(panel, "Class Summary", (PADDING, y), 0.56, TEXT_MUTED, 1)
    y += 18
    for class_id, class_name in CLASS_NAMES.items():
        color = CLASS_COLORS[class_id]
        draw_card(panel, PADDING, y, panel.shape[1] - PADDING * 2, 44)
        cv2.circle(panel, (PADDING + 22, y + 22), 7, color, -1)
        put_text(panel, CLASS_DISPLAY_NAMES[class_id], (PADDING + 42, y + 29), 0.55, TEXT, 1)
        put_text(panel, str(counts[class_id]), (panel.shape[1] - PADDING - 32, y + 30), 0.62, TEXT, 2)
        y += 54

    if panel.shape[0] >= 700:
        y += 12
        put_text(panel, "Latest Detections", (PADDING, y), 0.56, TEXT_MUTED, 1)
        y += 28
        for detection in detections[:4]:
            class_id = int(detection["class_id"])
            confidence = float(detection["confidence"])
            put_text(panel, f"{CLASS_DISPLAY_NAMES[class_id]}", (PADDING, y), 0.52, TEXT, 1)
            put_text(panel, f"{confidence:.2f}", (panel.shape[1] - PADDING - 48, y), 0.52, TEXT_MUTED, 1)
            y += 28
        if not detections:
            put_text(panel, "No objects detected", (PADDING, y), 0.52, TEXT_DIM, 1)

    put_text(panel, "Press q to quit", (PADDING, panel.shape[0] - 30), 0.52, TEXT_DIM, 1)


def compose_dashboard(
    frame: np.ndarray,
    fall_detected: bool,
    mode: str,
    detections: list[dict[str, object]],
    fps: float,
    display_width: int,
    conf: float,
) -> np.ndarray:
    scale = display_width / frame.shape[1]
    display_height = max(360, int(frame.shape[0] * scale))
    display_frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((display_height, display_width + PANEL_WIDTH, 3), dtype=np.uint8)
    canvas[:] = UI_BG
    canvas[:, :display_width] = display_frame
    draw_panel(canvas, display_width, fall_detected, mode, detections, fps, conf)
    return canvas


def run_one_step(frame: np.ndarray, model: YOLO, conf: float, imgsz: int) -> tuple[bool, list[dict[str, object]]]:
    result = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
    fall_detected = False
    detections = []
    if result.boxes is None:
        return False, detections
    for idx in range(len(result.boxes)):
        class_id = int(result.boxes.cls[idx].item())
        confidence = float(result.boxes.conf[idx].item())
        x1, y1, x2, y2 = result.boxes.xyxy[idx].cpu().numpy().astype(int).tolist()
        draw_detection(frame, class_id, confidence, (x1, y1, x2, y2))
        detections.append({"class_id": class_id, "confidence": confidence})
        fall_detected = fall_detected or class_id == 0
    return fall_detected, detections


def run_two_step(
    frame: np.ndarray,
    detector: YOLO,
    classifier: torch.nn.Module,
    transform: transforms.Compose,
    conf: float,
    imgsz: int,
    device: str,
) -> tuple[bool, list[dict[str, object]]]:
    result = detector.predict(frame, classes=[0], conf=conf, imgsz=imgsz, verbose=False)[0]
    fall_detected = False
    detections = []
    if result.boxes is None:
        return False, detections
    for idx in range(len(result.boxes)):
        x1, y1, x2, y2 = result.boxes.xyxy[idx].cpu().numpy().astype(int).tolist()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        class_id, confidence = classify_crop(classifier, transform, crop, device)
        draw_detection(frame, class_id, confidence, (x1, y1, x2, y2))
        detections.append({"class_id": class_id, "confidence": confidence})
        fall_detected = fall_detected or class_id == 0
    return fall_detected, detections


def open_source(source: str, camera_width: int, camera_height: int) -> cv2.VideoCapture:
    if source.isdigit():
        capture = cv2.VideoCapture(int(source))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        return capture
    return cv2.VideoCapture(source)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")


def main() -> None:
    args = parse_args()
    if args.mode == "one-step":
        require_file(args.one_step_weights, "one-step model weights")
        model = YOLO(str(args.one_step_weights))
        classifier = None
        transform = None
    else:
        require_file(args.person_weights, "person detector weights")
        require_file(args.classifier_weights, "two-step classifier weights")
        model = YOLO(str(args.person_weights))
        classifier = load_classifier(args.classifier_weights, args.device)
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    capture = open_source(args.source, args.camera_width, args.camera_height)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    writer = None
    last_frame_time = time.perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        current_time = time.perf_counter()
        fps = 1.0 / max(current_time - last_frame_time, 1e-6)
        last_frame_time = current_time

        if args.mode == "one-step":
            fall_detected, detections = run_one_step(frame, model, args.conf, args.imgsz)
        else:
            fall_detected, detections = run_two_step(
                frame,
                model,
                classifier,
                transform,
                args.conf,
                args.imgsz,
                args.device,
            )
        display_frame = compose_dashboard(
            frame,
            fall_detected,
            args.mode,
            detections,
            fps,
            args.display_width,
            args.conf,
        )

        if args.output is not None:
            if writer is None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                fps = capture.get(cv2.CAP_PROP_FPS)
                fps = fps if fps and fps > 0 else 20
                height, width = display_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))
            writer.write(display_frame)

        if not args.no_display:
            cv2.namedWindow("Fall Detection Demo", cv2.WINDOW_NORMAL)
            cv2.imshow("Fall Detection Demo", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
