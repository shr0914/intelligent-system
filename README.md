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

## Basic-Version Completion Scripts

After running the three notebooks, use these scripts to complete the remaining basic-version evidence.

### Condition-based analysis and report assets

```bash
.venv/bin/python scripts/evaluate_conditions.py
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

### Live GUI demo

One-step webcam demo:

```bash
./start_gui.sh
```

Two-step webcam demo:

```bash
./start_gui.sh two-step
```

You can also use the explicit launchers:

```bash
./start_one_step_gui.sh
./start_two_step_gui.sh
```

The GUI draws bounding boxes and predicted labels on the camera feed, while status, class counts, FPS, and alert information are shown in the side panel.

Headless annotated-video export:

```bash
./start_gui.sh --source path/to/demo.mp4 --no-display --output runs/report_assets/demo_one_step.mp4
```

## Environment Setup

Create a local virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

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
- basic OpenCV live GUI for one-step and two-step inference

Next:

- test the GUI with a webcam or recorded demo video
- add viewpoint/distance notes if the team wants a stronger condition analysis
- write the project report and record the presentation/demo video

## Important Project Notes

- The held-out test set must not be merged into the training split.
- Local helper materials for annotation and assistant context are intentionally excluded from Git tracking.
- The notebooks are intended to be readable for all team members and to support the final report workflow.
