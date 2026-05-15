# Basic Version Progress Summary

This document summarizes the current basic-version evidence after running the three notebooks plus:

```bash
.venv/bin/python scripts/evaluate_conditions.py
```

## Completed Basic Requirements

- One-step YOLO fall-detection architecture.
- Two-step person-detection plus crop-classification architecture.
- Held-out self-collected test set evaluation.
- One-step vs two-step comparison outputs.
- Lighting-based condition analysis using `Normal Light` and `Low Light`.
- Basic OpenCV live GUI with fall-alert banner.

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

## GUI Commands

One-step webcam demo:

```bash
.venv/bin/python scripts/live_fall_gui.py --mode one-step --source 0
```

Two-step webcam demo:

```bash
.venv/bin/python scripts/live_fall_gui.py --mode two-step --source 0
```

Headless annotated-video export:

```bash
.venv/bin/python scripts/live_fall_gui.py --mode one-step --source path/to/demo.mp4 --no-display --output runs/report_assets/demo_one_step.mp4
```

## Remaining Basic-Version Work

- Test the GUI with the actual webcam or final demo video source.
- Write the final 8-10 page report.
- Record the 10 minute presentation/demo video.
- Optionally add manual `viewpoint` and `distance` metadata for stronger condition analysis.
