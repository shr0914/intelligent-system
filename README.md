# Automated Fall Detection using AI Vision

Course project for **COS30018 - Intelligent Systems**.

This repository implements a computer-vision fall detection system with:

- **One-step YOLOv8 fall detection**
- **Two-step person detector + ResNet-18 action classifier**
- **Low-light robustness extension**
- **Pose-estimation extension**
- **Condition-based evaluation by lighting, viewpoint, distance, and environment**
- **OpenCV GUI for live prediction and fall alerts**

## Project Overview

The system detects three action classes:

| Class ID | Class Name |
|---:|---|
| 0 | fall detected |
| 1 | walk |
| 2 | sit |

The main comparison is between:

- **One-step model:** a single YOLOv8 model predicts bounding boxes and action classes directly.
- **Two-step model:** YOLOv8 first detects people, then a ResNet-18 classifier predicts the action from each person crop.

The project also includes research extensions for low-light enhancement and pose-estimation-based fall recognition.

## Repository Structure

```text
.
├── 01_one_step_baseline_pipeline.ipynb
├── 02_two_step_pipeline.ipynb
├── 03_model_comparison.ipynb
├── COS30018.pdf
├── requirements.txt
├── prepared_dataset/
├── test_dataset/
├── Test Dataset Distribution/
├── scripts/
│   ├── run_gui.py
│   ├── live_fall_gui.py
│   ├── low_light_enhancement.py
│   ├── pose_fall_extension.py
│   ├── run_report_assets.py
│   ├── evaluate_conditions.py
│   ├── evaluate_low_light_extension.py
│   ├── evaluate_pose_extension.py
│   └── tune_model_thresholds.py
└── runs/
    └── report_assets/
```

## Dataset

The working dataset is prepared in YOLO format:

```text
prepared_dataset/
  images/
    train/
    test/
  labels/
    train/
    test/
  data.yaml
```

The final evaluated test split contains **502 test images** and **648 labelled person instances**. Condition metadata is stored in:

```text
runs/report_assets/test_condition_metadata.csv
```

The condition metadata supports analysis by:

- lighting: bright, dim, low light, normal, shadowy
- viewpoint: front, back, side, angled, top, mixed
- distance: close, medium, far, very far, mixed
- environment: indoor and outdoor locations

## Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

If PyTorch installation fails, install the correct `torch` and `torchvision` packages for your OS/CUDA version from the official PyTorch selector, then rerun:

```bash
python3 -m pip install -r requirements.txt
```

## Running Order

Run the notebooks in this order:

```text
01_one_step_baseline_pipeline.ipynb
02_two_step_pipeline.ipynb
03_model_comparison.ipynb
```

The notebooks train/evaluate the models and generate the main comparison files used by the report scripts.

## Project Readiness Check

```bash
python3 scripts/check_project_ready.py
```

## Generate Report Assets

Run this after the notebooks have completed:

```bash
python3 scripts/run_report_assets.py --metadata runs/report_assets/test_condition_metadata.csv
```

Main outputs are saved to:

```text
runs/report_assets/
```

Important files include:

- `comparison_with_pose_summary.csv`
- `condition_summary.csv`
- `condition_summary_by_viewpoint.csv`
- `condition_summary_by_distance.csv`
- `condition_summary_by_environment.csv`
- `condition_per_class.csv`
- `condition_f1_by_lighting.png`
- `condition_f1_by_viewpoint.png`
- `condition_f1_by_distance.png`
- `condition_f1_by_environment.png`

## Low-Light Extension

Evaluate low-light enhancement modes:

```bash
python3 scripts/evaluate_low_light_extension.py --metadata runs/report_assets/test_condition_metadata.csv
```

Supported enhancement modes:

- `none`
- `gamma`
- `clahe`
- `gamma-clahe`
- `auto`

Outputs:

- `low_light_extension_summary.csv`
- `low_light_extension_per_class.csv`
- `low_light_extension_f1.png`
- `low_light_extension_examples/`

## Pose-Estimation Extension

Evaluate the pose-estimation extension:

```bash
python3 scripts/evaluate_pose_extension.py --metadata runs/report_assets/test_condition_metadata.csv
```

Evaluate pose estimation with low-light enhancement modes:

```bash
python3 scripts/evaluate_pose_extension.py --metadata runs/report_assets/test_condition_metadata.csv --low-light-only --enhancements none gamma clahe gamma-clahe auto
```

Outputs:

- `pose_metrics_summary.csv`
- `pose_per_class_metrics.csv`
- `pose_condition_summary.csv`
- `pose_low_light_enhancement_summary.csv`
- `pose_low_light_enhancement_f1.png`

## GUI Demo

The GUI supports webcam, video, image, and image-folder inputs.

### Webcam Modes

One-step YOLO:

```bash
python3 scripts/run_gui.py --mode one-step --source 0
```

Two-step detector + classifier:

```bash
python3 scripts/run_gui.py --mode two-step --source 0
```

Pose-estimation extension:

```bash
python3 scripts/run_gui.py --mode pose --source 0
```

### Low-Light GUI Mode

```bash
python3 scripts/run_gui.py --mode one-step --source 0 --enhancement auto
```

```bash
python3 scripts/run_gui.py --mode two-step --source 0 --enhancement auto
```

```bash
python3 scripts/run_gui.py --mode pose --source 0 --enhancement auto
```

### Useful GUI Options

```bash
python3 scripts/run_gui.py \
  --mode one-step \
  --source 0 \
  --enhancement auto \
  --fall-frames 3 \
  --alert-hold-seconds 2.5 \
  --log-csv runs/report_assets/gui_events.csv \
  --save-alert-frames runs/report_assets/gui_alert_frames
```

### Run On A Video File

```bash
python3 scripts/run_gui.py --mode one-step --source path/to/video.mp4
```

### Run On An Image Folder

```bash
python3 scripts/run_gui.py --mode one-step --source path/to/image_folder
```

### Headless Export

```bash
python3 scripts/run_gui.py \
  --mode one-step \
  --source path/to/video.mp4 \
  --no-display \
  --output runs/report_assets/demo_one_step.mp4 \
  --log-csv runs/report_assets/demo_one_step_events.csv
```

## Optional Threshold Tuning

Run optional inference threshold and crop-padding tuning:

```bash
python3 scripts/tune_model_thresholds.py --metadata runs/report_assets/test_condition_metadata.csv
```

Outputs:

- `tuned_inference_summary.csv`
- `tuned_inference_per_class.csv`
- `tuned_inference_match_details.csv`
- `tuned_inference_top_f1.png`

These tuning outputs are for analysis and experimentation. The main report comparison uses the final notebook and report-asset evaluation results.

## Final Result Summary

| Model / Evaluation | Main Result |
|---|---:|
| One-step YOLO overall F1 | 0.775 |
| One-step YOLO mAP@50 | 0.797 |
| Two-step detector + classifier overall F1 | 0.759 |
| Two-step crop classifier accuracy | 80.6% |
| Pose-rule extension overall F1 | 0.726 |
| One-step low-light auto-enhancement F1 | 0.810 |
| Two-step low-light auto-enhancement F1 | 0.797 |

## Notes

- The held-out test set must remain separate from the training split.
- Report-ready evaluation files are stored in `runs/report_assets/`.
- The GUI mode is selected at launch using `--mode one-step`, `--mode two-step`, or `--mode pose`.
- Low-light enhancement can be enabled using `--enhancement auto`.
- The submitted report is `COS30018.pdf`.
