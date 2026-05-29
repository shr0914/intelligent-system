from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms
from ultralytics import YOLO

from low_light_enhancement import ENHANCEMENT_MODES, enhance_low_light, mean_luminance
from pose_fall_extension import (
    PoseRuleConfig,
    draw_pose_prediction,
    predict_pose,
)


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
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PANEL_WIDTH = 420
PADDING = 24
UI_BG = (14, 16, 20)
PANEL_BG = (22, 25, 31)
PANEL_ACCENT = (36, 42, 52)
CARD_BG = (31, 36, 45)
CARD_BORDER = (61, 69, 84)
TEXT = (244, 246, 248)
TEXT_MUTED = (158, 166, 178)
TEXT_DIM = (116, 124, 138)
OK_GREEN = (84, 190, 120)
ALERT_RED = CLASS_COLORS[0]
REGULAR_FONT_CANDIDATES = [
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/Library/Fonts/Arial.ttf"),
]
BOLD_FONT_CANDIDATES = [
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    Path("C:/Windows/Fonts/segoeuib.ttf"),
    Path("C:/Windows/Fonts/arialbd.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    Path("/Library/Fonts/Arial Bold.ttf"),
]


@dataclass
class AlertResult:
    active: bool
    raw_fall: bool
    just_started: bool
    streak: int
    max_fall_confidence: float


class AlertState:
    def __init__(self, required_frames: int, hold_seconds: float, fall_confidence: float) -> None:
        self.required_frames = max(1, required_frames)
        self.hold_seconds = max(0.0, hold_seconds)
        self.fall_confidence = fall_confidence
        self.streak = 0
        self.active_until = 0.0
        self.was_active = False

    def update(self, detections: list[dict[str, object]], now: float) -> AlertResult:
        fall_confidences = [
            float(detection["confidence"])
            for detection in detections
            if int(detection["class_id"]) == 0 and float(detection["confidence"]) >= self.fall_confidence
        ]
        raw_fall = bool(fall_confidences)
        max_fall_confidence = max(fall_confidences, default=0.0)
        self.streak = self.streak + 1 if raw_fall else 0
        if self.streak >= self.required_frames:
            self.active_until = max(self.active_until, now + self.hold_seconds)
        active = now <= self.active_until
        just_started = active and not self.was_active
        self.was_active = active
        return AlertResult(active, raw_fall, just_started, self.streak, max_fall_confidence)


class FpsMeter:
    def __init__(self, window: int = 30) -> None:
        self.frame_times: deque[float] = deque(maxlen=window)
        self.last_time = time.perf_counter()

    def update(self) -> float:
        now = time.perf_counter()
        elapsed = max(now - self.last_time, 1e-6)
        self.last_time = now
        self.frame_times.append(elapsed)
        average_elapsed = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / max(average_elapsed, 1e-6)


@dataclass
class GuiSettings:
    confidence: float
    enhancement: str
    gamma: float
    clahe_clip_limit: float
    clahe_tile_grid_size: int
    low_light_threshold: float
    fall_frames: int
    alert_hold_seconds: float
    fall_confidence: float
    fall_confidence_tracks_confidence: bool
    logging_enabled: bool

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "GuiSettings":
        fall_confidence = args.conf if args.fall_conf is None else args.fall_conf
        return cls(
            confidence=args.conf,
            enhancement=args.enhancement,
            gamma=args.gamma,
            clahe_clip_limit=args.clahe_clip_limit,
            clahe_tile_grid_size=args.clahe_tile_grid_size,
            low_light_threshold=args.low_light_threshold,
            fall_frames=args.fall_frames,
            alert_hold_seconds=args.alert_hold_seconds,
            fall_confidence=fall_confidence,
            fall_confidence_tracks_confidence=args.fall_conf is None,
            logging_enabled=args.log_csv is not None,
        )


class FrameSource:
    def __init__(self, source: str, camera_width: int, camera_height: int, camera_fps: int) -> None:
        self.source = source
        self.capture: cv2.VideoCapture | None = None
        self.image_paths: list[Path] = []
        self.image_index = 0
        self.single_image: np.ndarray | None = None

        source_path = Path(source)
        if source.isdigit():
            self.capture = cv2.VideoCapture(int(source))
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
            self.capture.set(cv2.CAP_PROP_FPS, camera_fps)
        elif source_path.is_dir():
            self.image_paths = sorted(
                path for path in source_path.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
            )
        elif source_path.is_file() and source_path.suffix.lower() in IMAGE_SUFFIXES:
            self.single_image = cv2.imread(str(source_path))
        else:
            self.capture = cv2.VideoCapture(source)

    def is_opened(self) -> bool:
        if self.capture is not None:
            return self.capture.isOpened()
        if self.single_image is not None:
            return True
        return bool(self.image_paths)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.capture is not None:
            return self.capture.read()
        if self.single_image is not None:
            image = self.single_image.copy()
            self.single_image = None
            return True, image
        if self.image_index >= len(self.image_paths):
            return False, None
        image = cv2.imread(str(self.image_paths[self.image_index]))
        self.image_index += 1
        return image is not None, image

    def fps(self, fallback: float) -> float:
        if self.capture is None:
            return fallback
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else fallback

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live fall-detection GUI demo.")
    parser.add_argument(
        "--mode",
        choices=["one-step", "two-step", "pose"],
        default="one-step",
        help="Inference pipeline to run.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index, video path, image path, or image-folder path. Default: 0.",
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
    parser.add_argument(
        "--pose-weights",
        type=Path,
        default=PROJECT_ROOT / "yolov8n-pose.pt",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--enhancement",
        choices=ENHANCEMENT_MODES,
        default="none",
        help="Optional low-light enhancement mode applied before inference.",
    )
    parser.add_argument("--gamma", type=float, default=0.65)
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-tile-grid-size", type=int, default=8)
    parser.add_argument(
        "--low-light-threshold",
        type=float,
        default=90.0,
        help="Mean luminance threshold used by --enhancement auto.",
    )
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.10,
        help="Two-step crop padding ratio around detected person boxes before classification.",
    )
    parser.add_argument(
        "--two-step-fall-ratio-override",
        type=float,
        default=0.0,
        help=(
            "Demo-only two-step override. If greater than 0, relabel non-fall two-step "
            "predictions as Fall when the person crop width/height ratio is at least this value."
        ),
    )
    parser.add_argument(
        "--two-step-fall-override-conf",
        type=float,
        default=0.70,
        help="Confidence shown when --two-step-fall-ratio-override relabels a crop as Fall.",
    )
    parser.add_argument("--pose-fall-angle-threshold", type=float, default=55.0)
    parser.add_argument("--pose-fall-ratio-threshold", type=float, default=0.90)
    parser.add_argument("--pose-fall-vertical-spread-threshold", type=float, default=0.80)
    parser.add_argument("--pose-sit-ratio-threshold", type=float, default=0.45)
    parser.add_argument("--pose-sit-knee-hip-threshold", type=float, default=0.15)
    parser.add_argument("--pose-min-keypoint-conf", type=float, default=0.25)
    parser.add_argument("--pose-min-visible-keypoints", type=int, default=5)
    parser.add_argument(
        "--camera-width",
        type=int,
        default=1920,
        help="Requested webcam capture width.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=1080,
        help="Requested webcam capture height.",
    )
    parser.add_argument(
        "--camera-fps",
        type=int,
        default=60,
        help="Requested webcam capture FPS.",
    )
    parser.add_argument(
        "--canvas-width",
        type=int,
        default=1920,
        help="Output dashboard width.",
    )
    parser.add_argument(
        "--canvas-height",
        type=int,
        default=1080,
        help="Output dashboard height.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=60.0,
        help="Target display/output FPS for the GUI loop and exported videos.",
    )
    parser.add_argument(
        "--fall-frames",
        type=int,
        default=3,
        help="Consecutive fall frames needed before the alert activates.",
    )
    parser.add_argument(
        "--alert-hold-seconds",
        type=float,
        default=2.5,
        help="Seconds to keep the alert visible after it activates.",
    )
    parser.add_argument(
        "--fall-conf",
        type=float,
        default=None,
        help="Minimum fall confidence for alert logic. Defaults to --conf.",
    )
    parser.add_argument(
        "--log-csv",
        type=Path,
        default=None,
        help="Optional CSV path for frame-by-frame GUI event logging.",
    )
    parser.add_argument(
        "--save-alert-frames",
        type=Path,
        default=None,
        help="Optional folder for saving the first dashboard frame of each alert event.",
    )
    parser.add_argument(
        "--screenshot-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "report_assets" / "gui_screenshots",
        help="Folder used when pressing S to save the current dashboard frame.",
    )
    parser.add_argument(
        "--beep",
        action="store_true",
        help="Print a terminal bell when a new fall alert starts.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for smoke tests or short exports.",
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


def fit_frame_to_viewport(frame: np.ndarray, width: int, height: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized_width = max(1, int(frame.shape[1] * scale))
    resized_height = max(1, int(frame.shape[0] * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=interpolation)
    viewport = np.zeros((height, width, 3), dtype=np.uint8)
    viewport[:] = UI_BG
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    viewport[y : y + resized_height, x : x + resized_width] = resized
    return viewport, (x, y, resized_width, resized_height)


def draw_header_overlay(
    canvas: np.ndarray,
    viewport_box: tuple[int, int, int, int],
    mode: str,
    source: str,
    enhancement: str,
    luminance: float,
) -> None:
    x, y, width, _ = viewport_box
    header_height = 74
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + header_height), (10, 12, 16), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
    put_text(canvas, "Automated Fall Detection", (x + 26, y + 32), 0.78, TEXT, 2)
    subtitle = f"{mode.replace('-', ' ').title()} pipeline  |  enhancement: {enhancement}  |  luminance: {luminance:.1f}"
    put_text(canvas, subtitle, (x + 28, y + 58), 0.48, TEXT_MUTED, 1)
    source_text = source if len(source) <= 42 else f"...{source[-39:]}"
    text_width, _ = measure_text(source_text, 0.46, 1)
    put_text(canvas, source_text, (x + width - text_width - 26, y + 44), 0.46, TEXT_DIM, 1)


def draw_card(panel: np.ndarray, x: int, y: int, width: int, height: int, fill: tuple[int, int, int] = CARD_BG) -> None:
    cv2.rectangle(panel, (x, y), (x + width, y + height), fill, -1)
    cv2.rectangle(panel, (x, y), (x + width, y + height), CARD_BORDER, 1)


def draw_progress_bar(
    panel: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    value: float,
    color: tuple[int, int, int],
) -> None:
    value = max(0.0, min(1.0, value))
    cv2.rectangle(panel, (x, y), (x + width, y + height), (44, 49, 59), -1)
    cv2.rectangle(panel, (x, y), (x + int(width * value), y + height), color, -1)
    cv2.rectangle(panel, (x, y), (x + width, y + height), CARD_BORDER, 1)


def find_font_path(bold: bool = False) -> Path | None:
    candidates = BOLD_FONT_CANDIDATES if bold else REGULAR_FONT_CANDIDATES
    for path in candidates:
        if path.exists():
            return path
    return None


@lru_cache(maxsize=32)
def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = find_font_path(bold)
    if path is None:
        return ImageFont.load_default()
    return ImageFont.truetype(str(path), size=size)


def font_size_from_scale(scale: float) -> int:
    return max(11, int(round(scale * 34)))


def measure_text(text: str, scale: float, thickness: int = 1) -> tuple[int, int]:
    font = get_font(font_size_from_scale(scale), thickness > 1)
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 1,
) -> None:
    font = get_font(font_size_from_scale(scale), thickness > 1)
    text_width, text_height = measure_text(text, scale, thickness)
    x, baseline_y = origin
    top_y = baseline_y - text_height
    x0 = max(0, x - 4)
    y0 = max(0, top_y - 4)
    x1 = min(image.shape[1], x + text_width + 6)
    y1 = min(image.shape[0], baseline_y + 8)
    if x0 >= x1 or y0 >= y1:
        return

    roi = image[y0:y1, x0:x1]
    pil_image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    draw.text((x - x0, top_y - y0), text, font=font, fill=(color[2], color[1], color[0]))
    image[y0:y1, x0:x1] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def draw_panel(
    canvas: np.ndarray,
    panel_x: int,
    alert: AlertResult,
    mode: str,
    detections: list[dict[str, object]],
    fps: float,
    settings: GuiSettings,
    output_enabled: bool,
    log_enabled: bool,
    enhanced_frame: bool,
) -> None:
    panel = canvas[:, panel_x:]
    panel[:] = PANEL_BG
    cv2.line(canvas, (panel_x, 0), (panel_x, canvas.shape[0]), (62, 70, 84), 1)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 92), PANEL_ACCENT, -1)

    put_text(panel, "Live Status", (PADDING, 40), 0.82, TEXT, 2)
    put_text(panel, "Real-time fall recognition", (PADDING, 68), 0.48, TEXT_MUTED, 1)

    status_color = ALERT_RED if alert.active else OK_GREEN
    status_fill = (40, 42, 58) if alert.active else (30, 48, 40)
    status_text = "FALL ALERT" if alert.active else "NORMAL"
    draw_card(panel, PADDING, 116, panel.shape[1] - PADDING * 2, 132, status_fill)
    cv2.circle(panel, (PADDING + 32, 158), 13, status_color, -1)
    put_text(panel, status_text, (PADDING + 58, 169), 0.92, TEXT, 2)
    detail = "Alert locked" if alert.active else "Monitoring"
    if alert.raw_fall and not alert.active:
        detail = f"Confirming fall {min(alert.streak, settings.fall_frames)}/{settings.fall_frames}"
    put_text(panel, detail, (PADDING + 24, 210), 0.52, TEXT_MUTED, 1)
    draw_progress_bar(
        panel,
        PADDING + 24,
        226,
        panel.shape[1] - PADDING * 2 - 48,
        8,
        min(alert.streak / max(settings.fall_frames, 1), 1.0),
        status_color,
    )

    card_y = 276
    card_w = (panel.shape[1] - PADDING * 2 - 14) // 2
    metrics = [
        ("Objects", str(len(detections))),
        ("FPS", f"{fps:.1f}"),
    ]
    for index, (label, value) in enumerate(metrics):
        x = PADDING + index * (card_w + 14)
        draw_card(panel, x, card_y, card_w, 90)
        put_text(panel, label, (x + 16, card_y + 29), 0.48, TEXT_MUTED, 1)
        put_text(panel, value, (x + 16, card_y + 69), 0.84, TEXT, 2)

    draw_card(panel, PADDING, card_y + 112, panel.shape[1] - PADDING * 2, 78)
    put_text(panel, "Model", (PADDING + 16, card_y + 140), 0.48, TEXT_MUTED, 1)
    put_text(panel, f"{mode.replace('-', ' ').title()}", (PADDING + 88, card_y + 140), 0.52, TEXT, 1)
    put_text(panel, "Conf", (PADDING + 16, card_y + 172), 0.48, TEXT_MUTED, 1)
    put_text(panel, f"{settings.confidence:.2f}", (PADDING + 88, card_y + 172), 0.52, TEXT, 1)
    max_fall_text = f"Fall max {alert.max_fall_confidence:.2f}"
    text_width, _ = measure_text(max_fall_text, 0.48, 1)
    put_text(panel, max_fall_text, (panel.shape[1] - PADDING - text_width - 16, card_y + 172), 0.48, TEXT_MUTED, 1)
    enhancement_text = f"{settings.enhancement} {'on' if enhanced_frame else 'off'}"
    put_text(panel, "Enhance", (PADDING + 178, card_y + 140), 0.48, TEXT_MUTED, 1)
    put_text(panel, enhancement_text[:16], (PADDING + 258, card_y + 140), 0.48, TEXT, 1)
    tuning_y = card_y + 198
    put_text(panel, f"Gamma {settings.gamma:.2f}", (PADDING + 16, tuning_y), 0.44, TEXT_DIM, 1)
    put_text(panel, f"CLAHE {settings.clahe_clip_limit:.1f}", (PADDING + 142, tuning_y), 0.44, TEXT_DIM, 1)
    put_text(panel, f"Light {settings.low_light_threshold:.0f}", (PADDING + 272, tuning_y), 0.44, TEXT_DIM, 1)

    counts = {class_id: 0 for class_id in CLASS_NAMES}
    for detection in detections:
        counts[int(detection["class_id"])] += 1

    y = card_y + 224
    put_text(panel, "Class Summary", (PADDING, y), 0.56, TEXT_MUTED, 1)
    y += 18
    for class_id in CLASS_NAMES:
        color = CLASS_COLORS[class_id]
        draw_card(panel, PADDING, y, panel.shape[1] - PADDING * 2, 48)
        cv2.rectangle(panel, (PADDING, y), (PADDING + 5, y + 48), color, -1)
        cv2.circle(panel, (PADDING + 25, y + 24), 7, color, -1)
        put_text(panel, CLASS_DISPLAY_NAMES[class_id], (PADDING + 45, y + 31), 0.56, TEXT, 1)
        put_text(panel, str(counts[class_id]), (panel.shape[1] - PADDING - 34, y + 32), 0.64, TEXT, 2)
        y += 58

    y += 18
    if panel.shape[0] >= 760:
        put_text(panel, "Latest Detections", (PADDING, y), 0.56, TEXT_MUTED, 1)
        y += 30
        for detection in detections[:4]:
            class_id = int(detection["class_id"])
            confidence = float(detection["confidence"])
            color = CLASS_COLORS[class_id]
            cv2.circle(panel, (PADDING + 8, y - 5), 5, color, -1)
            put_text(panel, f"{CLASS_DISPLAY_NAMES[class_id]}", (PADDING + 22, y), 0.52, TEXT, 1)
            put_text(panel, f"{confidence:.2f}", (panel.shape[1] - PADDING - 52, y), 0.52, TEXT_MUTED, 1)
            y += 30
        if not detections:
            put_text(panel, "No objects detected", (PADDING, y), 0.52, TEXT_DIM, 1)

    footer_y = panel.shape[0] - 112
    draw_card(panel, PADDING, footer_y, panel.shape[1] - PADDING * 2, 78)
    put_text(panel, "Output", (PADDING + 16, footer_y + 25), 0.46, TEXT_MUTED, 1)
    output_text = "video" if output_enabled else "display"
    log_text = "log on" if log_enabled else "log off"
    put_text(panel, f"{output_text}  |  {log_text}", (PADDING + 92, footer_y + 25), 0.46, TEXT, 1)
    put_text(panel, "E mode  [/] gamma  -/= conf  C clahe", (PADDING + 16, footer_y + 47), 0.38, TEXT_DIM, 1)
    put_text(panel, "V light  F frames  H hold  L log  S shot  Q quit", (PADDING + 16, footer_y + 66), 0.34, TEXT_DIM, 1)


def compose_dashboard(
    frame: np.ndarray,
    alert: AlertResult,
    mode: str,
    detections: list[dict[str, object]],
    fps: float,
    canvas_width: int,
    canvas_height: int,
    settings: GuiSettings,
    output_enabled: bool,
    log_enabled: bool,
    source: str,
    enhanced_frame: bool,
    luminance: float,
) -> np.ndarray:
    panel_width = min(PANEL_WIDTH, max(360, canvas_width // 3))
    viewport_width = max(640, canvas_width - panel_width)
    viewport_height = max(480, canvas_height)
    display_frame, viewport_box = fit_frame_to_viewport(frame, viewport_width, viewport_height)
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[:] = UI_BG
    canvas[:, :viewport_width] = display_frame[:canvas_height, :viewport_width]
    draw_header_overlay(canvas, viewport_box, mode, source, settings.enhancement, luminance)
    draw_panel(
        canvas,
        viewport_width,
        alert,
        mode,
        detections,
        fps,
        settings,
        output_enabled,
        log_enabled,
        enhanced_frame,
    )
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
    crop_padding: float,
    fall_ratio_override: float,
    fall_override_conf: float,
) -> tuple[bool, list[dict[str, object]]]:
    result = detector.predict(frame, classes=[0], conf=conf, imgsz=imgsz, verbose=False)[0]
    fall_detected = False
    detections = []
    if result.boxes is None:
        return False, detections
    for idx in range(len(result.boxes)):
        x1, y1, x2, y2 = result.boxes.xyxy[idx].cpu().numpy().astype(int).tolist()
        box_width = x2 - x1
        box_height = y2 - y1
        pad_x = int(round(box_width * max(crop_padding, 0.0)))
        pad_y = int(round(box_height * max(crop_padding, 0.0)))
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(frame.shape[1], x2 + pad_x), min(frame.shape[0], y2 + pad_y)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        class_id, confidence = classify_crop(classifier, transform, crop, device)
        crop_width = max(1, x2 - x1)
        crop_height = max(1, y2 - y1)
        crop_ratio = crop_width / crop_height
        if fall_ratio_override > 0 and class_id != 0 and crop_ratio >= fall_ratio_override:
            class_id = 0
            confidence = max(confidence, fall_override_conf)
        draw_detection(frame, class_id, confidence, (x1, y1, x2, y2))
        detections.append({"class_id": class_id, "confidence": confidence})
        fall_detected = fall_detected or class_id == 0
    return fall_detected, detections


def run_pose(
    frame: np.ndarray,
    model: YOLO,
    conf: float,
    imgsz: int,
    config: PoseRuleConfig,
) -> tuple[bool, list[dict[str, object]]]:
    predictions = predict_pose(model, frame, conf, imgsz, config)
    detections = []
    fall_detected = False
    for prediction in predictions:
        draw_pose_prediction(frame, prediction)
        class_id = int(prediction["class_id"])
        confidence = float(prediction["score"])
        detections.append({"class_id": class_id, "confidence": confidence})
        fall_detected = fall_detected or class_id == 0
    return fall_detected, detections


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(
            f"Missing {label}: {path}\n"
            "Run the training notebooks first, or pass the correct path with the matching --*-weights option."
        )


def open_event_log(path: Path | None) -> tuple[object | None, csv.DictWriter | None]:
    if path is None:
        return None, None
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "timestamp",
            "frame",
            "mode",
            "objects",
            "detections",
            "raw_fall",
            "alert_active",
            "fall_streak",
            "max_fall_confidence",
            "fps",
            "enhancement",
            "enhancement_applied",
            "luminance",
        ],
    )
    writer.writeheader()
    return handle, writer


def write_event_log(
    writer: csv.DictWriter | None,
    frame_index: int,
    mode: str,
    detections: list[dict[str, object]],
    alert: AlertResult,
    fps: float,
    enhancement: str,
    enhancement_applied: bool,
    luminance: float,
) -> None:
    if writer is None:
        return
    detection_text = ";".join(
        f"{CLASS_DISPLAY_NAMES[int(detection['class_id'])]}:{float(detection['confidence']):.3f}"
        for detection in detections
    )
    writer.writerow(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "frame": frame_index,
            "mode": mode,
            "objects": len(detections),
            "detections": detection_text,
            "raw_fall": int(alert.raw_fall),
            "alert_active": int(alert.active),
            "fall_streak": alert.streak,
            "max_fall_confidence": f"{alert.max_fall_confidence:.4f}",
            "fps": f"{fps:.2f}",
            "enhancement": enhancement,
            "enhancement_applied": int(enhancement_applied),
            "luminance": f"{luminance:.2f}",
        }
    )


def save_alert_frame(directory: Path | None, dashboard: np.ndarray, frame_index: int) -> None:
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(str(directory / f"alert_{timestamp}_frame_{frame_index:06d}.jpg"), dashboard)


def save_manual_screenshot(directory: Path, dashboard: np.ndarray, frame_index: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = directory / f"screenshot_{timestamp}_frame_{frame_index:06d}.jpg"
    cv2.imwrite(str(path), dashboard)
    return path


def default_log_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "runs" / "report_assets" / f"gui_events_{timestamp}.csv"


def cycle_enhancement(current: str) -> str:
    modes = list(ENHANCEMENT_MODES)
    index = modes.index(current) if current in modes else 0
    return modes[(index + 1) % len(modes)]


def handle_key(
    key: int,
    settings: GuiSettings,
    alert_state: AlertState,
    display_frame: np.ndarray,
    frame_index: int,
    screenshot_dir: Path,
) -> bool:
    if key in (-1, 255):
        return False
    char = chr(key & 0xFF).lower()
    if char == "q":
        return True
    if char == "e":
        settings.enhancement = cycle_enhancement(settings.enhancement)
    elif char == "[":
        settings.gamma = max(0.20, settings.gamma - 0.05)
    elif char == "]":
        settings.gamma = min(1.50, settings.gamma + 0.05)
    elif char == "-":
        settings.confidence = max(0.05, settings.confidence - 0.05)
        if settings.fall_confidence_tracks_confidence:
            settings.fall_confidence = settings.confidence
    elif char in {"=", "+"}:
        settings.confidence = min(0.95, settings.confidence + 0.05)
        if settings.fall_confidence_tracks_confidence:
            settings.fall_confidence = settings.confidence
    elif char == "c":
        settings.clahe_clip_limit = 1.0 if settings.clahe_clip_limit >= 4.0 else settings.clahe_clip_limit + 0.5
    elif char == "v":
        settings.low_light_threshold = 50.0 if settings.low_light_threshold >= 140.0 else settings.low_light_threshold + 10.0
    elif char == "f":
        settings.fall_frames = 1 if settings.fall_frames >= 6 else settings.fall_frames + 1
    elif char == "h":
        settings.alert_hold_seconds = 1.0 if settings.alert_hold_seconds >= 5.0 else settings.alert_hold_seconds + 0.5
    elif char == "l":
        settings.logging_enabled = not settings.logging_enabled
    elif char == "s":
        path = save_manual_screenshot(screenshot_dir, display_frame, frame_index)
        print(f"Saved screenshot: {path}")

    alert_state.required_frames = settings.fall_frames
    alert_state.hold_seconds = settings.alert_hold_seconds
    alert_state.fall_confidence = settings.fall_confidence
    return False


def main() -> None:
    args = parse_args()
    settings = GuiSettings.from_args(args)
    alert_state = AlertState(settings.fall_frames, settings.alert_hold_seconds, settings.fall_confidence)

    pose_config = PoseRuleConfig(
        fall_angle_threshold=args.pose_fall_angle_threshold,
        fall_ratio_threshold=args.pose_fall_ratio_threshold,
        fall_vertical_spread_threshold=args.pose_fall_vertical_spread_threshold,
        sit_ratio_threshold=args.pose_sit_ratio_threshold,
        sit_knee_hip_threshold=args.pose_sit_knee_hip_threshold,
        min_keypoint_conf=args.pose_min_keypoint_conf,
        min_visible_keypoints=args.pose_min_visible_keypoints,
    )

    if args.mode == "one-step":
        require_file(args.one_step_weights, "one-step model weights")
        model = YOLO(str(args.one_step_weights))
        classifier = None
        transform = None
    elif args.mode == "two-step":
        require_file(args.person_weights, "person detector weights")
        require_file(args.classifier_weights, "two-step classifier weights")
        model = YOLO(str(args.person_weights))
        classifier = load_classifier(args.classifier_weights, args.device)
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    else:
        model = YOLO(str(args.pose_weights))
        classifier = None
        transform = None

    source = FrameSource(args.source, args.camera_width, args.camera_height, args.camera_fps)
    if not source.is_opened():
        raise SystemExit(
            f"Could not open source: {args.source}\n"
            "Use a webcam index such as 0, or pass a valid video, image, or image-folder path."
        )

    writer = None
    log_handle, log_writer = open_event_log(args.log_csv)
    fps_meter = FpsMeter()
    frame_index = 0
    window_created = False
    while True:
        frame_start = time.perf_counter()
        ok, frame = source.read()
        if not ok or frame is None:
            break
        frame_index += 1
        fps = fps_meter.update()
        luminance = mean_luminance(frame)
        inference_frame, enhancement_applied = enhance_low_light(
            frame,
            mode=settings.enhancement,
            gamma=settings.gamma,
            clahe_clip_limit=settings.clahe_clip_limit,
            clahe_tile_grid_size=settings.clahe_tile_grid_size,
            low_light_threshold=settings.low_light_threshold,
        )

        if args.mode == "one-step":
            fall_detected, detections = run_one_step(inference_frame, model, settings.confidence, args.imgsz)
        elif args.mode == "two-step":
            fall_detected, detections = run_two_step(
                inference_frame,
                model,
                classifier,
                transform,
                settings.confidence,
                args.imgsz,
                args.device,
                args.crop_padding,
                args.two_step_fall_ratio_override,
                args.two_step_fall_override_conf,
            )
        else:
            fall_detected, detections = run_pose(
                inference_frame,
                model,
                settings.confidence,
                args.imgsz,
                pose_config,
            )
        _ = fall_detected
        alert = alert_state.update(detections, time.perf_counter())
        display_frame = compose_dashboard(
            inference_frame,
            alert,
            args.mode,
            detections,
            fps,
            args.canvas_width,
            args.canvas_height,
            settings,
            args.output is not None,
            settings.logging_enabled,
            args.source,
            enhancement_applied,
            luminance,
        )
        if settings.logging_enabled and log_writer is None:
            log_handle, log_writer = open_event_log(args.log_csv or default_log_path())
        if settings.logging_enabled:
            write_event_log(
                log_writer,
                frame_index,
                args.mode,
                detections,
                alert,
                fps,
                settings.enhancement,
                enhancement_applied,
                luminance,
            )
        if alert.just_started:
            save_alert_frame(args.save_alert_frames, display_frame, frame_index)
            if args.beep:
                print("\a", end="", flush=True)

        if args.output is not None:
            if writer is None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                output_fps = source.fps(args.target_fps)
                output_fps = min(max(output_fps, 1), args.target_fps)
                height, width = display_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(args.output), fourcc, output_fps, (width, height))
            writer.write(display_frame)

        if not args.no_display:
            if not window_created:
                cv2.namedWindow("Fall Detection Demo", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Fall Detection Demo", args.canvas_width, args.canvas_height)
                window_created = True
            cv2.imshow("Fall Detection Demo", display_frame)
            elapsed_ms = (time.perf_counter() - frame_start) * 1000
            frame_delay_ms = max(1, int((1000 / max(args.target_fps, 1)) - elapsed_ms))
            key = cv2.waitKey(frame_delay_ms) & 0xFF
            if handle_key(key, settings, alert_state, display_frame, frame_index, args.screenshot_dir):
                break
        if args.max_frames is not None and frame_index >= args.max_frames:
            break

    source.release()
    if writer is not None:
        writer.release()
    if log_handle is not None:
        log_handle.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
