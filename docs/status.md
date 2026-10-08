# Development status

## Implemented

- Isolated Python 3.11 Conda environment and dedicated Jupyter kernel.
- Matched CPU-only PyTorch 2.10.0 / Torchvision 0.25.0, with dependency lock file.
- Seven sanitized experiment notebooks, reusable Python ML modules, tests, and setup documentation.
- Image integrity and duplicate audits, conservative filename-group splitting, deterministic manifest.
- RGB/letterbox preprocessing, conservative augmentation, bounded-memory DataLoaders.
- Pretrained one-logit ResNet18, frozen-head training, optional layer4 fine-tuning.
- Progress bars, epoch timing, peak process memory, validation metrics, early stopping.
- Atomic checkpoints with optimizer/scheduler/random state and interrupted-epoch recovery.
- Held-out evaluation utilities with explicit selection lock and undefined-metric handling.
- Real gradient-based Grad-CAM and shared checkpoint-backed inference.
- Smoke-model isolation: normal inference and final evaluation refuse smoke artifacts.

## Dataset audit

The selected local extraction contains 5,856 images. Its nested copy is byte-identical.
No unreadable images were found. There are 62 image rows in decoded-pixel duplicate groups
and 170 filename-derived groups crossing original splits. There is no authoritative patient
metadata; filename grouping remains a documented heuristic.

The development manifest preserves all 624 original test images. It excludes 457 development
images linked to test groups and 22 further repeated pixel images, retaining paths and reasons.
The resulting training set has 3,787 images (1,071 NORMAL / 2,716 PNEUMONIA), and validation
has 950 images (269 NORMAL / 681 PNEUMONIA). The original 16 validation images remain an
unused auxiliary set. No test predictions have been generated.

## Verification results

- `pip check`: no broken requirements. Installed package inventory contains no NVIDIA,
  CUDA, ROCm, or Triton packages.
- Eight ML tests pass, including exact equality of resumed and uninterrupted model weights.
- Ruff lint and formatting checks pass.
- All seven notebooks execute successfully. Notebooks 01–03 perform actual exploration,
  preprocessing, and pretrained-model checks; notebook 04 executes the opt-in smoke test.
  Notebooks 05–07 verify locked/missing-trained-model states without evaluating test data.
- Real-data smoke run: one epoch, 32 training / 16 validation images, batch size 8,
  image size 224, two CPU threads, zero loader workers. Training took 3.19 seconds and
  validation 1.65 seconds; total epoch time was 4.87 seconds. Peak process RSS was 595.33 MiB
  (this excludes the desktop, other processes, and the OS file cache).
- Reloaded checkpoint logits match exactly. Real-validation-image Grad-CAM produced finite
  values in [0, 1]; inference plus explanation took 0.44 seconds in this check. Normal inference
  correctly refuses the smoke artifact.
- Executed notebooks and smoke outputs remain in ignored local `artifacts/` directories.

Linear extrapolation of the tiny smoke run suggests roughly eight minutes per full head epoch,
or forty minutes for five epochs. This is only a planning estimate; image dimensions, caching,
desktop load, storage, and early stopping can change elapsed time substantially. No full run
has been started, and no final evaluation metrics are available.

## Remaining phases

1. Obtain explicit approval for full training, then train and lock model/threshold selection.
2. Run the held-out evaluation once and review actual errors and Grad-CAM examples.
3. Implement the FastAPI upload/history API with safe content validation and model-unavailable states.
4. Set up Node.js and implement the Next.js dashboard, analysis, history, performance, and methodology pages.
5. Run API/frontend/integration tests and document actual local startup commands/screenshots.
6. Publish a source-only snapshot to the requested private GitHub repository when repository-creation
   tooling/authentication is available. The existing differently named repository was verified private.

The API and dashboard are not yet implemented. Full training, held-out evaluation, deployment,
and publication have not been performed. The current phase ends at a verified CPU smoke run,
as required before full training approval.
