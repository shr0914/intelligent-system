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

Next:

- refine the two-step evaluation and comparison
- perform condition-based analysis such as low-light vs normal-light
- implement the GUI for live webcam inference and fall alerting

## Important Project Notes

- The held-out test set must not be merged into the training split.
- Local helper materials for annotation and assistant context are intentionally excluded from Git tracking.
- The notebooks are intended to be readable for all team members and to support the final report workflow.
