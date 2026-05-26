# Marking Audit Against COS30018 Project 3-2

Date checked: 2026-05-26

This audit compares the current repository against the assignment requirements and marking scheme. The project already has most of the technical work needed for a high mark, but the final grade will depend heavily on packaging the evidence clearly in the report, video, and final submission zip.

## Quick Verdict

Current technical status is strong:

- One-step YOLO fall detector exists and has held-out test evaluation outputs.
- Two-step detector plus crop classifier exists and has end-to-end evaluation outputs.
- Self-collected held-out test set exists and exceeds 50 images per class overall.
- GUI exists and runs in headless smoke mode with local weights.
- Lighting condition analysis exists.
- Low-light extension assets exist.
- Pose-rule extension assets exist.

Main full-grade risks:

- Final report is not written yet.
- Presentation/demo video is not recorded yet.
- Notebooks `02_two_step_pipeline.ipynb` and `03_model_comparison.ipynb` currently have no saved outputs.
- `viewpoint`, `distance`, and `environment` metadata columns are still blank, so condition analysis is mainly lighting-based.
- `runs/`, datasets, and model weights are ignored by Git, so the final Canvas zip must explicitly include the working artifacts.
- Research component should be framed as one approved extension, not as scattered extra scripts.

## Verified Evidence

Readiness command used:

```bash
.venv/bin/python scripts/check_project_ready.py
```

Result: passed.

Dataset counts from the readiness check:

| Split | Count |
| --- | ---: |
| prepared train images | 10,466 |
| prepared train labels | 10,466 |
| prepared test images | 274 |
| prepared test labels | 274 |
| raw self-collected test images | 276 |

Raw self-collected test image distribution:

| Lighting | Class | Count |
| --- | --- | ---: |
| Low Light | Fall | 39 |
| Low Light | Sit | 51 |
| Low Light | Walk | 43 |
| Normal Light | Fall | 30 |
| Normal Light | Sit | 42 |
| Normal Light | Walk | 71 |

Overall by class:

| Class | Count |
| --- | ---: |
| Fall | 69 |
| Sit | 93 |
| Walk | 114 |

GUI smoke command used:

```bash
.venv/bin/python scripts/run_gui.py --mode one-step --source test_dataset --no-display --max-frames 3 --output /tmp/fall_gui_smoke.mp4
```

Result: passed and produced `/tmp/fall_gui_smoke.mp4`.

## Marking Scheme Audit

| Requirement | Marks | Current Status | Risk | Fix for Full Marks |
| --- | ---: | --- | --- | --- |
| Task 1: data collection, one-step architecture, new dataset, condition analysis | 20 | Mostly complete. One-step notebook, trained YOLO weights, held-out test metrics, prediction images, and lighting condition analysis exist. Test set exceeds 50 images per class overall. | Medium. Condition analysis currently proves lighting but not viewpoint/distance/environment. Notebook has only some saved outputs. | Fill metadata for viewpoint, distance, and environment, rerun report assets, save executed notebook outputs, and show sample annotations/data collection process in report. |
| Task 2: two-step architecture, new dataset, compare with one-step, condition analysis | 30 | Mostly complete. Two-step pipeline uses YOLO person detector plus ResNet18 crop classifier. End-to-end precision/recall/F1 and comparison CSVs exist. | Medium. Two-step notebook has zero saved outputs. Comparison uses F1 for two-step but mAP for one-step, so explain metric differences carefully. | Re-execute/save notebook outputs, include end-to-end IoU matching explanation, include condition tables, and discuss why two-step performs better in current F1 comparison. |
| GUI: live prediction and fall alert | 10 | Complete in code. `scripts/run_gui.py` supports one-step, two-step, and pose modes, webcam/video/folder inputs, alert banner, logging, and video export. Headless smoke run passed. | Low to medium. Need actual webcam/final demo test and visible proof in video/report. | Record GUI section in final video showing live prediction and visible alert. Save one screenshot and one CSV/video evidence asset. |
| Project report | 10 | Not done. | High. This is an automatic lost 10 if missing. | Write 8-10 page report with all required sections and video link. Use generated plots/tables from `runs/report_assets/`. |
| Presentation/demo video | 10 | Not done. | High. This is an automatic lost 10 if missing. | Record a 10-minute video with architecture, dataset, demo, metrics, extension, limitations, and team contribution. |
| Research component | up to 40 | Strong candidate work exists: low-light enhancement extension and pose-rule extension. | Medium. Rubric says choose one extension and get tutor approval. If approval/story is unclear, marks may drop. | Present low-light robustness as the main approved extension. Optionally mention pose as additional analysis, but keep the research narrative focused. |
| Programming practice penalty | up to -20 | Code is separated into scripts, notebooks, docs, and reproducible commands. | Low to medium. Some notebooks lack outputs; `README.md` still says GUI/extension work is future in one section despite being done later. | Clean README status wording, execute notebooks, keep final zip organized, and avoid submitting stale duplicated runs unless needed. |
| Weekly progress penalty | up to -50 | Git history shows incremental commits. | Unknown. Depends on tutor demos. | Prepare a short progress timeline from commits and `docs/basic_version_progress.md`. |

## Current Core Metrics To Use

Condition analysis:

| Model | Lighting | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| one-step YOLO | Low Light | 0.654 | 0.643 | 0.648 |
| one-step YOLO | Normal Light | 0.665 | 0.751 | 0.706 |
| two-step detector + classifier | Low Light | 0.797 | 0.841 | 0.818 |
| two-step detector + classifier | Normal Light | 0.776 | 0.828 | 0.801 |

Comparison summary:

| Model | Precision | Recall | F1 | mAP50 | mAP50-95 | Crop Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| one-step YOLO | 0.635 | 0.689 | not reported | 0.688 | 0.380 | n/a |
| two-step detector + classifier | 0.786 | 0.834 | 0.809 | n/a | n/a | 0.839 |
| pose-rule extension | 0.735 | 0.803 | 0.768 | n/a | n/a | n/a |

Low-light extension:

| Model | Best Enhancement | Baseline F1 | Best F1 |
| --- | --- | ---: | ---: |
| one-step YOLO | auto | 0.580 | 0.661 |
| two-step detector + classifier | clahe | 0.821 | 0.823 |

Important interpretation:

- The strongest extension story is that low-light enhancement helps the one-step detector substantially, from F1 0.580 to 0.661.
- Two-step is already robust in low light, so enhancement only slightly improves it.
- The two-step model performs better than one-step in the current F1-style comparison, especially under low light.
- `sit` is a weaker class than `walk` and should be discussed as a limitation.

## Things To Fix Before Submission

1. Write the final report.
   Include cover page, TOC, introduction, architecture, data collection/annotation, ML techniques, scenarios/examples, critical analysis, practical application, conclusion, video link, and contribution table.

2. Record the 10-minute video.
   Show the actual GUI with webcam or a recorded video source, trigger a fall alert, show model comparison metrics, and explain the low-light extension.

3. Fill condition metadata.
   Edit `runs/report_assets/test_condition_metadata.csv` or create a cleaner copy with `viewpoint`, `distance`, and `environment` filled. Then rerun:

   ```bash
   .venv/bin/python scripts/run_report_assets.py --metadata runs/report_assets/test_condition_metadata.csv
   ```

4. Save executed notebook outputs.
   Open and run `02_two_step_pipeline.ipynb` and `03_model_comparison.ipynb`, or at least run their non-training summary/evaluation cells and save outputs. Current saved output counts are:

   | Notebook | Saved Outputs |
   | --- | ---: |
   | `01_one_step_baseline_pipeline.ipynb` | 5 |
   | `02_two_step_pipeline.ipynb` | 0 |
   | `03_model_comparison.ipynb` | 0 |

5. Clean README status wording.
   The README has later sections showing GUI/extensions are done, but an early paragraph still says GUI and extension work will be added later. Make the status consistent.

6. Build a final submission zip carefully.
   Git ignores `runs/`, datasets, and `*.pt`, but the assignment asks for code and a working system. Include at least:

   - notebooks
   - scripts
   - `requirements.txt`
   - final report
   - model weights needed by GUI
   - minimal demo/test data or clear included sample data
   - report assets used in the report
   - instructions to run the demo

7. Confirm GitHub access.
   Remote exists at `https://github.com/shr0914/intelligent-system.git`, but confirm the tutor/lecturer has read-only access.

8. Choose the research component story.
   Use low-light robustness as the primary research extension because it directly matches the assignment example and has clear experimental results. Mention pose estimation only as optional supporting work unless the tutor approved it as the main extension.

## Suggested Report Structure

Use this as the 8-10 page plan:

1. Cover page and team contribution table.
2. Introduction and problem motivation.
3. System architecture with one-step, two-step, GUI, and extension diagram.
4. Data collection and annotation, including provided dataset and self-collected test set counts.
5. One-step YOLO method and training/evaluation setup.
6. Two-step YOLO person detector plus ResNet18 crop classifier method.
7. Results and comparison under lighting, viewpoint, distance, and environment.
8. GUI prototype and alert mechanism.
9. Low-light robustness extension and results.
10. Critical analysis, practical applications, limitations, and conclusion.

## Suggested 10-Minute Video Plan

| Time | Content |
| --- | --- |
| 0:00-0:45 | Problem, assignment goal, team roles |
| 0:45-1:45 | Dataset collection and annotation |
| 1:45-3:00 | One-step architecture |
| 3:00-4:15 | Two-step architecture |
| 4:15-5:30 | Metrics comparison and condition analysis |
| 5:30-6:45 | Low-light extension |
| 6:45-8:30 | Live GUI demo with alert |
| 8:30-9:30 | Limitations and improvements |
| 9:30-10:00 | Conclusion |

## Estimated Grade Position

If submitted exactly as-is without report/video, the project would lose at least 20 marks from the basic 80-mark component. With the final report and video completed, the current technical implementation should be capable of a high basic-component mark.

For full-grade ambition, the most important remaining work is not more modeling. It is making the evidence undeniable: executed notebooks, richer condition metadata, a polished report, a clear demo video, and a working submission package.
