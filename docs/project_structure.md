# Project Structure

This project keeps source code, datasets, model outputs, and report assets separate so the final submission stays easier to explain.

## Main Files

```text
README.md
requirements.txt
01_one_step_baseline_pipeline.ipynb
02_two_step_pipeline.ipynb
03_model_comparison.ipynb
COS30018 - Assignment 2026_S1 - Project 3-2.pdf
```

## Main Commands

Use direct Python commands:

```bash
python scripts/check_project_ready.py
python scripts/run_report_assets.py
python scripts/evaluate_low_light_extension.py
python scripts/evaluate_pose_extension.py
python scripts/run_gui.py --mode pose --source 0
```

## Source Code

```text
scripts/
  check_project_ready.py
  evaluate_conditions.py
  evaluate_low_light_extension.py
  evaluate_pose_extension.py
  live_fall_gui.py
  low_light_enhancement.py
  pose_fall_extension.py
  run_gui.py
  run_report_assets.py
  tune_model_thresholds.py
```

## Data And Generated Outputs

These folders are intentionally not committed:

```text
fall_dataset/
prepared_dataset/
test_dataset/
runs/
labeling/
.venv/
*.pt
```

## Report Assets To Use

```text
runs/report_assets/
  condition_summary.csv
  condition_per_class.csv
  low_light_extension_summary.csv
  pose_metrics_summary.csv
  pose_condition_summary.csv
  comparison_with_pose_summary.csv
```
