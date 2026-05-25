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

The current repository focuses on the model-building and evaluation pipeline first, with the GUI and extension work to be added after the initial analysis is stable.

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

### 1. One-Step Detection Pipeline

Notebook:

- `01_one_step_baseline_pipeline.ipynb`

Main tasks:

- verify environment and dataset
- train the one-step detection model
- evaluate on the held-out test set
- save metrics and prediction outputs

### 2. Two-Step Pipeline

Notebook:

- `02_two_step_pipeline.ipynb`

Main tasks:

- create person crops from YOLO annotations
- train a crop classifier
- build the detector + classifier pipeline
- evaluate the two-step system on the held-out test set

### 3. Model Comparison

Notebook:

- `03_model_comparison.ipynb`

Main tasks:

- compare one-step and two-step outputs
- summarize metrics and per-class behavior
- generate report-ready tables and plots



## Basic-Version Completion Scripts

After running the three notebooks, use these scripts to complete the remaining basic-version evidence.

### Condition-based analysis and report assets

Cross-platform command:

```bash
python scripts/run_report_assets.py
```

Linux/macOS:

```bash
./generate_report_assets.sh
```

Windows PowerShell:

```powershell
.\generate_report_assets.ps1
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

The exported `runs/report_assets/test_condition_metadata.csv` also includes optional columns for `viewpoint`, `distance`, and `environment`. If those columns are filled and passed back to the script, extra condition summaries are generated:

Cross-platform:

```bash
python scripts/run_report_assets.py --metadata runs/report_assets/test_condition_metadata.csv
```

Linux/macOS:

```bash
./generate_report_assets.sh --metadata runs/report_assets/test_condition_metadata.csv
```

Windows PowerShell:

```powershell
.\generate_report_assets.ps1 --metadata runs/report_assets/test_condition_metadata.csv
```

Readiness check only:

```bash
python scripts/check_project_ready.py
```

### Live GUI demo

One-step webcam demo:

Cross-platform:

```bash
python scripts/run_gui.py --mode one-step --source 0
```

Linux/macOS:

```bash
./start_gui.sh
```

Windows PowerShell:

```powershell
.\start_gui.ps1
```

Two-step webcam demo:

Cross-platform:

```bash
python scripts/run_gui.py --mode two-step --source 0
```

Linux/macOS:

```bash
./start_gui.sh two-step
```

Windows PowerShell:

```powershell
.\start_gui.ps1 two-step
```

You can also use the explicit launchers:

Linux/macOS:

```bash
./start_one_step_gui.sh
./start_two_step_gui.sh
```

Windows PowerShell:

```powershell
.\start_one_step_gui.ps1
.\start_two_step_gui.ps1
```

The GUI draws bounding boxes and predicted labels on the camera feed, while status, class counts, FPS, and alert information are shown in the side panel.

Headless annotated-video export:

Cross-platform:

```bash
python scripts/run_gui.py --mode one-step --source path/to/demo.mp4 --no-display --output runs/report_assets/demo_one_step.mp4
```

Linux/macOS:

```bash
./start_gui.sh --source path/to/demo.mp4 --no-display --output runs/report_assets/demo_one_step.mp4
```

Windows PowerShell:

```powershell
.\start_gui.ps1 --source path\to\demo.mp4 --no-display --output runs\report_assets\demo_one_step.mp4
```

## Environment Setup

Create a local virtual environment and install dependencies:

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch installation fails, install the matching `torch`, `torchvision`, and `torchaudio` packages for your OS and CUDA version from the official PyTorch install selector, then rerun `pip install -r requirements.txt`.

## Current Status

Completed:

- dataset preparation
- self-collected test-set annotation and correction
- one-step model training
- one-step held-out evaluation
- two-step pipeline implementation
- initial one-step vs two-step comparison
- lighting-based condition analysis
- report-ready condition tables and plots
- basic OpenCV live GUI for one-step and two-step inference
- short reproducible commands for GUI launch, report asset generation, and readiness checks

Next:

- optionally fill viewpoint/distance/environment metadata for stronger condition analysis
- write the project report and record the presentation/demo video

## Important Project Notes

- The held-out test set must not be merged into the training split.
- Local helper materials for annotation and assistant context are intentionally excluded from Git tracking.
- The notebooks are intended to be readable for all team members and to support the final report workflow.
