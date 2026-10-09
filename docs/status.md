# Development status

## Completed

- CPU-only Python environment, Jupyter kernel, and ResNet18 pipeline.
- Dataset checks, duplicate detection, and a deterministic split manifest.
- Five approved head-training epochs; no fine-tuning.
- One held-out evaluation, model export, Grad-CAM, and inference notebooks.
- FastAPI uploads, matching model reports, and optional SQLite metadata history.
- Next.js overview, analysis, history, performance, and methodology pages.
- Development branch merged into the existing private repository’s `main` branch.
  All 11,720 original dataset and metadata entries are preserved.

## Results

Epoch 5 was selected by validation ROC-AUC (0.99280), with the threshold fixed at 0.5.
On the 624-image test set: accuracy 87.66%, precision 85.81%, recall 96.15%,
specificity 73.50%, F1 90.69%, and ROC-AUC 0.94486. There were 15 false negatives
and 62 false positives. The test results were not used to retune the model.

The five completed epochs took 42.0 minutes in total, excluding interrupted work and
setup. Peak process memory was about 606 MiB. The final epoch resumed from the saved
epoch-4 checkpoint; the notebook now uses one guarded call for fresh or resumed training.

## Verification

- 21 Python tests passed, including exact interrupted-run recovery and API validation.
- Six frontend tests passed with a single thread worker. Fork-worker startup timed out
  on this machine, so the test configuration uses threads.
- Python lint, formatting, dependency checks, TypeScript, and frontend lint passed.
- The production frontend build passed; dependency audit reported no vulnerabilities.
- Notebooks 05–07 executed with the final model; all source notebooks validate and compile.
- Two real validation images passed through the frontend API proxy, including Grad-CAM.
- All four report plots matched the exported model and loaded successfully.
- Navigation, measured results, methodology, and the history-disabled page were checked.
- The analysis page displayed a real prediction and all explanation images at normal
  and 390-pixel viewport widths, without horizontal page overflow.

Model version: `resnet18-9ae7d62a87b9`. Local artifacts include the model, checkpoints,
reports, executed notebooks, and integration-check results. History remains off by default.

## Limits

The model is for education and research, not patient care. Scores are uncalibrated,
no validated INCONCLUSIVE band is enabled, and filename-derived groups cannot establish
patient independence. The dataset is pediatric and limited in acquisition setting.
External validation and calibration remain future work.
