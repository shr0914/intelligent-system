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

### Low-light robustness extension

The extension evaluates inference-time low-light enhancement using:

- `none`
- `gamma`
- `clahe`
- `gamma-clahe`
- `auto`

Generate extension assets:

Cross-platform:

```bash
python scripts/evaluate_low_light_extension.py
```

Linux/macOS:

```bash
./generate_low_light_extension.sh
```

Windows PowerShell:

```powershell
.\generate_low_light_extension.ps1
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

Cross-platform:

```bash
python scripts/run_gui.py --mode one-step --source 0 --enhancement auto
python scripts/run_gui.py --mode two-step --source 0 --enhancement clahe
```

Linux/macOS:

```bash
./start_gui.sh --enhancement auto
./start_gui.sh two-step --enhancement clahe
```

Windows PowerShell:

```powershell
.\start_gui.ps1 --enhancement auto
.\start_gui.ps1 two-step --enhancement clahe
```

### Inference tuning

The current two-step classifier already uses a pretrained ResNet18, so the fastest model-quality improvement is tuning inference settings and crop padding rather than retraining immediately.

Generate threshold and padding tuning assets:

Cross-platform:

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

The default GUI canvas is Full HD (`1920x1080`) and requests `1920x1080` camera input at `60` FPS when the camera supports it. The fall alert is smoothed by default: it activates after `3` consecutive fall frames and remains visible briefly to avoid flickering. The two-step GUI uses `0.10` crop padding by default because tuning improved the end-to-end F1 score.

Runtime keyboard controls:

| Key | Action |
| --- | --- |
| `E` | Cycle enhancement mode: `none`, `gamma`, `clahe`, `gamma-clahe`, `auto` |
| `[` / `]` | Decrease / increase gamma |
| `-` / `=` | Decrease / increase confidence threshold |
| `C` | Cycle CLAHE clip limit |
| `V` | Cycle low-light threshold |
| `F` | Cycle required fall frames |
| `H` | Cycle alert hold duration |
| `L` | Toggle CSV event logging |
| `S` | Save current dashboard screenshot |
| `Q` | Quit |

Useful GUI options:

Cross-platform:

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

Linux/macOS:

```bash
./start_gui.sh \
  --enhancement auto \
  --fall-frames 3 \
  --alert-hold-seconds 2.5 \
  --log-csv runs/report_assets/gui_events.csv \
  --save-alert-frames runs/report_assets/gui_alert_frames
```

Windows PowerShell:

```powershell
.\start_gui.ps1 `
  --enhancement auto `
  --fall-frames 3 `
  --alert-hold-seconds 2.5 `
  --log-csv runs\report_assets\gui_events.csv `
  --save-alert-frames runs\report_assets\gui_alert_frames
```

Image and folder sources are also supported for smoke tests:

Cross-platform:

```bash
python scripts/run_gui.py --mode one-step --source test_dataset --no-display --max-frames 30
```

Linux/macOS:

```bash
./start_gui.sh --source test_dataset --no-display --max-frames 30
```

Windows PowerShell:

```powershell
.\start_gui.ps1 --source test_dataset --no-display --max-frames 30
```

Headless annotated-video export:

Cross-platform:

```bash
python scripts/run_gui.py --mode one-step --source path/to/demo.mp4 --no-display --output runs/report_assets/demo_one_step.mp4 --log-csv runs/report_assets/demo_one_step_events.csv
```

Linux/macOS:

```bash
./start_gui.sh --source path/to/demo.mp4 --no-display --output runs/report_assets/demo_one_step.mp4 --log-csv runs/report_assets/demo_one_step_events.csv
```

Windows PowerShell:

```powershell
.\start_gui.ps1 --source path\to\demo.mp4 --no-display --output runs\report_assets\demo_one_step.mp4 --log-csv runs\report_assets\demo_one_step_events.csv
```

### Gradio web app

The project also includes an optional browser-based Gradio interface. It keeps the native OpenCV GUI as the lowest-latency fallback, while providing a cleaner web UI with native-like live OpenCV camera detection, browser webcam streaming, and image upload.

Cross-platform:

```bash
python scripts/run_web_app.py
```

Linux/macOS:

```bash
./start_web_app.sh
```

Windows PowerShell:

```powershell
.\start_web_app.ps1
```

Then open:

```text
http://127.0.0.1:7860
```

Useful options:

```bash
python scripts/run_web_app.py --host 0.0.0.0 --port 7860
```

For faster live-camera updates, lower the stream interval if your machine can keep up:

```bash
python scripts/run_web_app.py --stream-every 0.2
```

The web app supports:

- one-step and two-step inference
- native-like live detection from an OpenCV camera loop
- image upload and browser webcam streaming
- low-light enhancement modes
- confidence threshold control
- gamma / CLAHE / low-light threshold controls
- two-step crop padding
- dashboard output with detections and alert status

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
- one-step baseline training
- one-step held-out evaluation
- two-step baseline implementation
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
