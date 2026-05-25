# Automated Fall Detection using AI Vision

Course project for `COS30018 - Intelligent Systems`.

This repository implements and compares two fall-detection architectures:

- one-step fall detection
- two-step fall detection

The project uses a labeled training dataset together with a self-collected held-out test set covering both normal-light and low-light conditions.

## Project Scope

The assignment requires:

- a one-step fall detection model
- a two-step fall detection model
- comparative analysis under different conditions
- a GUI with live prediction and fall alerting

The current repository focuses on the model-building and evaluation pipeline first, with the GUI and extension work to be added after the baseline analysis is stable.

## Repository Structure

```text
.
├── COS30018 - Assignment 2026_S1 - Project 3-2.pdf
├── 01_one_step_baseline_pipeline.ipynb
├── 02_two_step_pipeline.ipynb
├── 03_model_comparison.ipynb
├── requirements.txt
├── fall_dataset/
├── prepared_dataset/
└── test_dataset/
```

## Dataset Layout

The working dataset used by the notebooks is:

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

Notes:

- `train/` contains the provided labeled dataset prepared for training
- `test/` contains the self-collected and corrected held-out evaluation set
- the test split is kept separate from training throughout the project

## Notebook Workflow

### 1. One-step baseline

Notebook:

- `01_one_step_baseline_pipeline.ipynb`

Main tasks:

- verify environment and dataset
- train the one-step YOLO baseline
- evaluate on the held-out test set
- save metrics and prediction outputs

### 2. Two-step pipeline

Notebook:

- `02_two_step_pipeline.ipynb`

Main tasks:

- create person crops from YOLO annotations
- train a crop classifier
- build the detector + classifier pipeline
- evaluate the two-step system on the held-out test set

### 3. Model comparison

Notebook:

- `03_model_comparison.ipynb`

Main tasks:

- compare one-step and two-step outputs
- summarize metrics and per-class behavior
- generate report-ready tables and plots

## Running The Project

Run the notebooks in order:

```text
01_one_step_baseline_pipeline.ipynb
02_two_step_pipeline.ipynb
03_model_comparison.ipynb
```

Then use the Python scripts below for checks, report assets, extensions, and demos.

### Readiness Check

```bash
python scripts/check_project_ready.py
```

### Condition Analysis

```bash
python scripts/run_report_assets.py
```

This exports lighting-based analysis to:

```text
runs/report_assets/
  condition_summary.csv
  condition_per_class.csv
  condition_match_details.csv
  test_condition_metadata.csv
  condition_f1_by_lighting.png
  condition_precision_by_lighting.png
  condition_recall_by_lighting.png
  report_asset_notes.json
```

The current condition split uses the folder structure in `test_dataset/`:

- `Normal Light`
- `Low Light`

The exported `runs/report_assets/test_condition_metadata.csv` includes optional columns for `viewpoint`, `distance`, and `environment`. If those columns are filled, pass the CSV back to generate extra condition summaries:

```bash
python scripts/run_report_assets.py --metadata runs/report_assets/test_condition_metadata.csv
```

### Low-Light Robustness Extension

The low-light extension evaluates inference-time enhancement using:

- `none`
- `gamma`
- `clahe`
- `gamma-clahe`
- `auto`

Generate extension assets:

```bash
python scripts/evaluate_low_light_extension.py
```

This exports:

```text
runs/report_assets/
  low_light_extension_summary.csv
  low_light_extension_per_class.csv
  low_light_extension_match_details.csv
  low_light_extension_f1.png
  low_light_extension_notes.json
  low_light_extension_examples/
```

The GUI can use the same enhancement modes:

```bash
python scripts/run_gui.py --mode one-step --source 0 --enhancement auto
python scripts/run_gui.py --mode two-step --source 0 --enhancement clahe
python scripts/run_gui.py --mode pose --source 0 --enhancement auto
```

### Pose Estimation Extension

The pose extension adds a third method that uses YOLO human keypoints and transparent posture rules instead of only bounding-box appearance. It extracts features such as torso angle, width-to-height ratio, visible keypoint count, and hip/knee layout, then predicts:

- `fall detected`
- `walk`
- `sit`

Generate pose extension assets:

```bash
python scripts/evaluate_pose_extension.py
```

To combine pose estimation with the low-light enhancement pipeline, run the pose evaluator on low-light images with all enhancement modes:

```bash
python scripts/evaluate_pose_extension.py --low-light-only --enhancements none gamma clahe gamma-clahe auto
```

This exports:

```text
runs/pose_extension/
  pose_metrics_summary.csv
  pose_per_class_metrics.csv
  pose_match_details.csv
  pose_predictions.csv
  pose_condition_summary.csv
  pose_condition_per_class.csv
  pose_f1_by_lighting.png
  pose_low_light_enhancement_summary.csv
  pose_low_light_enhancement_per_class.csv
  pose_low_light_enhancement_match_details.csv
  pose_low_light_enhancement_f1.png
  pose_extension_notes.json
  examples/
```

It also copies report-ready pose summaries to:

```text
runs/report_assets/
  pose_metrics_summary.csv
  pose_per_class_metrics.csv
  pose_condition_summary.csv
  pose_condition_per_class.csv
  pose_match_details.csv
  pose_f1_by_lighting.png
  pose_low_light_enhancement_summary.csv
  pose_low_light_enhancement_per_class.csv
  pose_low_light_enhancement_match_details.csv
  pose_low_light_enhancement_f1.png
  pose_extension_notes.json
  comparison_with_pose_summary.csv
```

### Inference Tuning

The current two-step classifier already uses a pretrained ResNet18, so the fastest model-quality improvement is tuning inference settings and crop padding rather than retraining immediately.

```bash
python scripts/tune_model_thresholds.py
```

This exports:

```text
runs/report_assets/
  tuned_inference_summary.csv
  tuned_inference_per_class.csv
  tuned_inference_match_details.csv
  tuned_inference_top_f1.png
```

Current best tuned settings:

| Model | Best setting | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| one-step YOLO | `conf=0.25` | 0.660 | 0.701 | 0.680 |
| two-step detector + classifier | `person_conf=0.30`, `padding=0.10` | 0.797 | 0.862 | 0.828 |

### Live GUI Demo

```bash
python scripts/run_gui.py --mode one-step --source 0
python scripts/run_gui.py --mode two-step --source 0
python scripts/run_gui.py --mode pose --source 0
```

The GUI draws bounding boxes or pose skeletons on the camera feed, while status, class counts, FPS, and alert information are shown in the side panel.

Useful GUI options:

```bash
python scripts/run_gui.py \
  --mode one-step \
  --source 0 \
  --enhancement auto \
  --crop-padding 0.10 \
  --fall-frames 3 \
  --alert-hold-seconds 2.5 \
  --log-csv runs/report_assets/gui_events.csv \
  --save-alert-frames runs/report_assets/gui_alert_frames
```

Image and folder sources are also supported for smoke tests:

```bash
python scripts/run_gui.py --mode one-step --source test_dataset --no-display --max-frames 30
```

Headless annotated-video export:

```bash
python scripts/run_gui.py --mode one-step --source path/to/demo.mp4 --no-display --output runs/report_assets/demo_one_step.mp4 --log-csv runs/report_assets/demo_one_step_events.csv
```

## Environment Setup

Create a local virtual environment and install dependencies:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PyTorch installation fails, install the matching `torch`, `torchvision`, and `torchaudio` packages for your OS and CUDA version from the official PyTorch install selector, then rerun `pip install -r requirements.txt`.

## Current Status

Completed:

- dataset preparation
- self-collected test-set annotation and correction
- one-step baseline training
- one-step held-out evaluation
- two-step baseline implementation
- initial one-step vs two-step comparison
- lighting-based condition analysis
- report-ready condition tables and plots
- basic OpenCV live GUI for one-step, two-step, and pose inference
- pose-estimation extension with keypoint features, posture rules, lighting split, and skeleton examples
- short reproducible commands for GUI launch, report asset generation, and readiness checks

Next:

- optionally fill viewpoint/distance/environment metadata for stronger condition analysis
- write the project report and record the presentation/demo video

## Important Project Notes

- The held-out test set must not be merged into the training split.
- Local helper materials for annotation and assistant context are intentionally excluded from Git tracking.
- The notebooks are intended to be readable for all team members and to support the final report workflow.
