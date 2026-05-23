from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from torchvision import transforms
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from live_fall_gui import (
    CLASS_DISPLAY_NAMES,
    AlertState,
    compose_dashboard,
    load_classifier,
    run_one_step,
    run_two_step,
)
from low_light_enhancement import ENHANCEMENT_MODES, enhance_low_light, mean_luminance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONE_STEP_WEIGHTS = PROJECT_ROOT / "runs" / "baseline_runs" / "one_step_baseline" / "weights" / "best.pt"
DEFAULT_PERSON_WEIGHTS = PROJECT_ROOT / "yolov8n.pt"
DEFAULT_CLASSIFIER_WEIGHTS = PROJECT_ROOT / "runs" / "two_step" / "classifier_runs" / "best_classifier.pt"


@dataclass
class WebRuntime:
    alert_state: AlertState


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio web app for fall-detection inference.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--one-step-weights", type=Path, default=DEFAULT_ONE_STEP_WEIGHTS)
    parser.add_argument("--person-weights", type=Path, default=DEFAULT_PERSON_WEIGHTS)
    parser.add_argument("--classifier-weights", type=Path, default=DEFAULT_CLASSIFIER_WEIGHTS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--canvas-width", type=int, default=1280)
    parser.add_argument("--canvas-height", type=int, default=720)
    parser.add_argument("--stream-every", type=float, default=0.5)
    return parser.parse_args(argv)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(
            f"Missing {label}: {path}\n"
            "Run the training notebooks first, or pass the correct model path."
        )


@lru_cache(maxsize=1)
def get_one_step_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


@lru_cache(maxsize=1)
def get_person_detector(weights_path: str) -> YOLO:
    return YOLO(weights_path)


@lru_cache(maxsize=1)
def get_classifier(weights_path: str, device: str) -> torch.nn.Module:
    return load_classifier(Path(weights_path), device)


def detections_to_rows(detections: list[dict[str, object]]) -> list[list[object]]:
    return [
        [
            CLASS_DISPLAY_NAMES.get(int(detection["class_id"]), str(detection["class_id"])),
            round(float(detection["confidence"]), 3),
        ]
        for detection in detections
    ]


def count_summary(detections: list[dict[str, object]]) -> str:
    counts = {class_id: 0 for class_id in CLASS_DISPLAY_NAMES}
    for detection in detections:
        counts[int(detection["class_id"])] += 1
    return " | ".join(f"{CLASS_DISPLAY_NAMES[class_id]}: {counts[class_id]}" for class_id in counts)


def predict_image(
    image_rgb: np.ndarray | None,
    mode: str,
    confidence: float,
    enhancement: str,
    gamma: float,
    clahe_clip_limit: float,
    low_light_threshold: float,
    crop_padding: float,
    fall_frames: int,
    alert_hold_seconds: float,
    one_step_weights: str,
    person_weights: str,
    classifier_weights: str,
    device: str,
    canvas_width: int,
    canvas_height: int,
    runtime: WebRuntime | None,
) -> tuple[np.ndarray | None, str, str, list[list[object]], WebRuntime]:
    if runtime is None:
        runtime = WebRuntime(AlertState(fall_frames, alert_hold_seconds, confidence))
    runtime.alert_state.required_frames = max(1, int(fall_frames))
    runtime.alert_state.hold_seconds = max(0.0, float(alert_hold_seconds))
    runtime.alert_state.fall_confidence = float(confidence)

    if image_rgb is None:
        return None, "No image provided.", "", [], runtime

    frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    luminance = mean_luminance(frame_bgr)
    inference_frame, enhancement_applied = enhance_low_light(
        frame_bgr,
        mode=enhancement,
        gamma=gamma,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid_size=8,
        low_light_threshold=low_light_threshold,
    )

    started = time.perf_counter()
    if mode == "one-step":
        model = get_one_step_model(one_step_weights)
        _, detections = run_one_step(inference_frame, model, confidence, 640)
    else:
        detector = get_person_detector(person_weights)
        classifier = get_classifier(classifier_weights, device)
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        _, detections = run_two_step(
            inference_frame,
            detector,
            classifier,
            transform,
            confidence,
            640,
            device,
            crop_padding,
        )

    fps = 1.0 / max(time.perf_counter() - started, 1e-6)
    alert = runtime.alert_state.update(detections, time.perf_counter())
    dashboard_bgr = compose_dashboard(
        inference_frame,
        alert,
        mode,
        detections,
        fps,
        canvas_width,
        canvas_height,
        settings=type(
            "WebSettings",
            (),
            {
                "confidence": confidence,
                "enhancement": enhancement,
                "gamma": gamma,
                "clahe_clip_limit": clahe_clip_limit,
                "low_light_threshold": low_light_threshold,
                "fall_frames": fall_frames,
            },
        )(),
        output_enabled=False,
        log_enabled=False,
        source="Gradio webcam/upload",
        enhanced_frame=enhancement_applied,
        luminance=luminance,
    )
    dashboard_rgb = cv2.cvtColor(dashboard_bgr, cv2.COLOR_BGR2RGB)
    status = "FALL ALERT" if alert.active else "NORMAL"
    detail = (
        f"Status: {status}\n"
        f"Mode: {mode}\n"
        f"Enhancement: {enhancement} ({'applied' if enhancement_applied else 'not applied'})\n"
        f"Luminance: {luminance:.1f}\n"
        f"FPS: {fps:.1f}"
    )
    return dashboard_rgb, detail, count_summary(detections), detections_to_rows(detections), runtime


def build_app(args: argparse.Namespace) -> gr.Blocks:
    require_file(args.one_step_weights, "one-step model weights")
    require_file(args.person_weights, "person detector weights")
    require_file(args.classifier_weights, "two-step classifier weights")

    with gr.Blocks(title="Fall Detection Web Demo") as app:
        gr.Markdown("## Automated Fall Detection Web Demo")
        runtime_holder: dict[str, WebRuntime | None] = {"runtime": None}
        with gr.Row():
            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.Tab("Live OpenCV camera"):
                        camera_index = gr.Number(value=0, precision=0, label="Camera index")
                        with gr.Row():
                            start_camera_button = gr.Button("Start Live Detection", variant="primary")
                            stop_camera_button = gr.Button("Stop")
                    with gr.Tab("Browser webcam"):
                        webcam_image = gr.Image(
                            label="Browser webcam stream",
                            sources=["webcam"],
                            type="numpy",
                            streaming=True,
                        )
                    with gr.Tab("Image upload"):
                        upload_image = gr.Image(
                            label="Upload image",
                            sources=["upload"],
                            type="numpy",
                        )
                        run_button = gr.Button("Run Prediction", variant="primary")
            with gr.Column(scale=3):
                mode = gr.Radio(["one-step", "two-step"], value="one-step", label="Model")
                confidence = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence")
                enhancement = gr.Dropdown(list(ENHANCEMENT_MODES), value="auto", label="Low-light enhancement")
                gamma = gr.Slider(0.20, 1.50, value=0.65, step=0.05, label="Gamma")
                clahe_clip_limit = gr.Slider(1.0, 4.0, value=2.0, step=0.5, label="CLAHE clip limit")
                low_light_threshold = gr.Slider(50, 140, value=90, step=5, label="Low-light threshold")
                crop_padding = gr.Slider(0.0, 0.30, value=0.10, step=0.05, label="Two-step crop padding")
                fall_frames = gr.Slider(1, 6, value=3, step=1, label="Fall frames")
                alert_hold_seconds = gr.Slider(1.0, 5.0, value=2.5, step=0.5, label="Alert hold seconds")

        with gr.Row():
            output_image = gr.Image(label="Annotated dashboard", type="numpy")
        with gr.Row():
            status = gr.Textbox(label="Status", lines=6, elem_id="status_box")
            summary = gr.Textbox(label="Class counts")
        detections = gr.Dataframe(headers=["Class", "Confidence"], label="Detections")

        def run_prediction(
            image,
            mode,
            confidence,
            enhancement,
            gamma,
            clahe_clip_limit,
            low_light_threshold,
            crop_padding,
            fall_frames,
            alert_hold_seconds,
        ):
            dashboard, detail, counts, rows, runtime = predict_image(
                image,
                mode,
                confidence,
                enhancement,
                gamma,
                clahe_clip_limit,
                low_light_threshold,
                crop_padding,
                int(fall_frames),
                alert_hold_seconds,
                str(args.one_step_weights),
                str(args.person_weights),
                str(args.classifier_weights),
                args.device,
                args.canvas_width,
                args.canvas_height,
                runtime_holder["runtime"],
            )
            runtime_holder["runtime"] = runtime
            return dashboard, detail, counts, rows

        def live_camera_loop(
            camera_index,
            mode,
            confidence,
            enhancement,
            gamma,
            clahe_clip_limit,
            low_light_threshold,
            crop_padding,
            fall_frames,
            alert_hold_seconds,
        ):
            camera_id = int(camera_index or 0)
            capture = cv2.VideoCapture(camera_id)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.canvas_width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.canvas_height)
            capture.set(cv2.CAP_PROP_FPS, 60)

            if not capture.isOpened():
                yield None, f"Could not open camera index {camera_id}.", "", []
                return

            runtime = WebRuntime(AlertState(int(fall_frames), alert_hold_seconds, confidence))
            try:
                while True:
                    loop_started = time.perf_counter()
                    ok, frame_bgr = capture.read()
                    if not ok:
                        yield None, f"Could not read from camera index {camera_id}.", "", []
                        return

                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    dashboard, detail, counts, rows, runtime = predict_image(
                        frame_rgb,
                        mode,
                        confidence,
                        enhancement,
                        gamma,
                        clahe_clip_limit,
                        low_light_threshold,
                        crop_padding,
                        int(fall_frames),
                        alert_hold_seconds,
                        str(args.one_step_weights),
                        str(args.person_weights),
                        str(args.classifier_weights),
                        args.device,
                        args.canvas_width,
                        args.canvas_height,
                        runtime,
                    )
                    yield dashboard, detail, counts, rows

                    elapsed = time.perf_counter() - loop_started
                    if elapsed < args.stream_every:
                        time.sleep(args.stream_every - elapsed)
            finally:
                capture.release()

        def stop_live_camera():
            return None

        control_inputs = [
            mode,
            confidence,
            enhancement,
            gamma,
            clahe_clip_limit,
            low_light_threshold,
            crop_padding,
            fall_frames,
            alert_hold_seconds,
        ]
        outputs = [output_image, status, summary, detections]
        upload_inputs = [upload_image, *control_inputs]
        webcam_inputs = [webcam_image, *control_inputs]
        live_camera_inputs = [camera_index, *control_inputs]

        live_event = start_camera_button.click(
            fn=live_camera_loop,
            inputs=live_camera_inputs,
            outputs=outputs,
            show_progress="hidden",
        )
        stop_camera_button.click(
            fn=stop_live_camera,
            inputs=None,
            outputs=None,
            cancels=[live_event],
            show_progress="hidden",
        )
        run_button.click(
            fn=run_prediction,
            inputs=upload_inputs,
            outputs=outputs,
        )
        webcam_image.stream(
            fn=run_prediction,
            inputs=webcam_inputs,
            outputs=outputs,
            show_progress="hidden",
            trigger_mode="always_last",
            concurrency_limit=1,
            stream_every=args.stream_every,
        )
    return app


def main() -> None:
    args = parse_args()
    app = build_app(args)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
