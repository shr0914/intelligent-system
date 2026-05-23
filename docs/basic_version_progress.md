# Basic Version Progress Summary

This document summarizes the current basic-version evidence after running the three notebooks plus:

```bash
python scripts/run_report_assets.py
```

## Completed Basic Requirements

- One-step YOLO fall-detection architecture.
- Two-step person-detection plus crop-classification architecture.
- Held-out self-collected test set evaluation.
- One-step vs two-step comparison outputs.
- Lighting-based condition analysis using `Normal Light` and `Low Light`.
- Basic OpenCV live GUI with Full HD dashboard, smoothed fall-alert banner, optional CSV event logging, and alert-frame export.
- Reproducible readiness check for dataset counts and expected outputs.

## Lighting-Based Condition Summary

The fixed-threshold condition analysis uses:

- IoU threshold: `0.5`
- One-step confidence threshold: `0.25`
- Prepared test images: `274`
- Raw self-collected test images: `276`

| Model | Lighting | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| one-step YOLO | Low Light | 0.654 | 0.643 | 0.648 |
| one-step YOLO | Normal Light | 0.665 | 0.751 | 0.706 |
| two-step detector + classifier | Low Light | 0.797 | 0.841 | 0.818 |
| two-step detector + classifier | Normal Light | 0.776 | 0.828 | 0.801 |

## Per-Class Lighting Summary

| Model | Lighting | Class | Precision | Recall | F1 |
| --- | --- | --- | ---: | ---: | ---: |
| one-step YOLO | Low Light | fall detected | 0.673 | 0.660 | 0.667 |
| one-step YOLO | Low Light | walk | 0.750 | 0.646 | 0.694 |
| one-step YOLO | Low Light | sit | 0.532 | 0.623 | 0.574 |
| one-step YOLO | Normal Light | fall detected | 0.507 | 0.756 | 0.607 |
| one-step YOLO | Normal Light | walk | 0.715 | 0.949 | 0.816 |
| one-step YOLO | Normal Light | sit | 0.769 | 0.455 | 0.571 |
| two-step detector + classifier | Low Light | fall detected | 0.872 | 0.820 | 0.845 |
| two-step detector + classifier | Low Light | walk | 0.796 | 0.937 | 0.860 |
| two-step detector + classifier | Low Light | sit | 0.731 | 0.717 | 0.724 |
| two-step detector + classifier | Normal Light | fall detected | 0.844 | 0.600 | 0.701 |
| two-step detector + classifier | Normal Light | walk | 0.778 | 1.000 | 0.875 |
| two-step detector + classifier | Normal Light | sit | 0.738 | 0.727 | 0.733 |

## Generated Report Assets

The script exports local report-ready files under `runs/report_assets/`:

- `condition_summary.csv`
- `condition_per_class.csv`
- `condition_match_details.csv`
- `test_condition_metadata.csv`
- `condition_f1_by_lighting.png`
- `condition_precision_by_lighting.png`
- `condition_recall_by_lighting.png`
- `report_asset_notes.json`

To regenerate and verify these assets:

Cross-platform:

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

The generated `test_condition_metadata.csv` includes optional `viewpoint`, `distance`, and `environment` columns. After filling those values, rerun:

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

This creates additional summaries such as `condition_summary_by_viewpoint.csv`, `condition_summary_by_distance.csv`, and matching F1 charts when those columns contain values.

## Low-Light Extension Assets

The extension script evaluates inference-time low-light enhancement modes:

```bash
python scripts/evaluate_low_light_extension.py
```

Generated assets:

- `low_light_extension_summary.csv`
- `low_light_extension_per_class.csv`
- `low_light_extension_match_details.csv`
- `low_light_extension_f1.png`
- `low_light_extension_notes.json`
- `low_light_extension_examples/`

Current low-light summary:

| Model | Best enhancement | Baseline F1 | Best F1 |
| --- | --- | ---: | ---: |
| one-step YOLO | auto | 0.580 | 0.661 |
| two-step detector + classifier | clahe | 0.821 | 0.823 |

## Inference Tuning Assets

The tuning script searches practical inference settings without collecting new data:

```bash
python scripts/tune_model_thresholds.py
```

Generated assets:

- `tuned_inference_summary.csv`
- `tuned_inference_per_class.csv`
- `tuned_inference_match_details.csv`
- `tuned_inference_top_f1.png`

Current best settings:

| Model | Best setting | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| one-step YOLO | `conf=0.25` | 0.660 | 0.701 | 0.680 |
| two-step detector + classifier | `person_conf=0.30`, `padding=0.10` | 0.797 | 0.862 | 0.828 |

## GUI Commands

One-step webcam demo:

```bash
python scripts/run_gui.py --mode one-step --source 0
```

Two-step webcam demo:

```bash
python scripts/run_gui.py --mode two-step --source 0
```

Headless annotated-video export:

```bash
python scripts/run_gui.py --mode one-step --source path/to/demo.mp4 --no-display --output runs/report_assets/demo_one_step.mp4 --log-csv runs/report_assets/demo_one_step_events.csv
```

The live GUI also supports runtime keyboard controls for demo use:

- `E`: cycle enhancement mode.
- `[` / `]`: adjust gamma.
- `-` / `=`: adjust confidence threshold.
- `C`: cycle CLAHE clip limit.
- `V`: cycle low-light threshold.
- `F`: cycle required fall frames.
- `H`: cycle alert hold duration.
- `L`: toggle CSV event logging.
- `S`: save a dashboard screenshot.
- `Q`: quit.

## Gradio Web App

Optional browser interface:

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

Open `http://127.0.0.1:7860`. The web app supports native-like live detection from an OpenCV camera loop, image upload, browser webcam streaming, one-step/two-step inference, low-light enhancement controls, confidence tuning, crop padding, dashboard output, and detection summaries. Use `--stream-every 0.2` for faster live-camera refresh when the machine can keep up.

## Remaining Basic-Version Work

- Test the GUI with the actual webcam or final demo video source.
- Write the final 8-10 page report.
- Record the 10 minute presentation/demo video.
- Optionally add manual `viewpoint` and `distance` metadata for stronger condition analysis.
