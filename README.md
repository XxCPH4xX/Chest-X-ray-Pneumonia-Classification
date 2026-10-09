# Chest X-ray Pneumonia Classification

A local, CPU-only research workflow for binary NORMAL/PNEUMONIA classification with
ImageNet-pretrained ResNet18, notebook experiments, reproducible splits, resumable training,
and gradient-based visualizations. This is an educational research project, not a clinically
validated diagnostic system.

## Current implementation

Seven Jupyter notebooks cover dataset exploration, preprocessing, model development, training,
evaluation, Grad-CAM, and inference. Reusable functions live in Python modules.

The FastAPI backend and Next.js dashboard are implemented. Four of five approved training
epochs are saved; the run needs to resume before final evaluation and model export. The app
shows a model-unavailable state until an evaluated model is exported. Smoke checkpoints are
rejected by normal inference and test evaluation. See [development status](docs/status.md).

## Hardware and environment

Target: Fedora 44, Intel Core i5-4590S (4 cores), 12 GB RAM, Intel integrated graphics.
Execution is explicitly CPU-only throughout. There is no accelerator detection or automatic
device selection. No CUDA, NVIDIA, ROCm, or GPU-specific packages are required.

Python 3.11 is isolated in Conda environment `chest-xray-cpu`; the base and `dev` environments
are not modified. PyTorch `2.10.0+cpu` and Torchvision `0.25.0+cpu` use the matching pair
documented by [PyTorch](https://pytorch.org/get-started/previous-versions/#v2100), from its
CPU wheel index. The resolved remaining dependencies are recorded in `requirements-lock.txt`.

Fresh installation, from the project root:

```bash
conda env create -f environment.yml
conda activate chest-xray-cpu
python -m pip install -r requirements-cpu.txt
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m ipykernel install --user --name chest-xray-cpu --display-name 'Chest X-ray (CPU)'
python -m pip check
python -m ml.runtime
```

The environment and kernel are already installed on the development machine. To work:

```bash
conda activate chest-xray-cpu
jupyter lab --ip=127.0.0.1
```

Select **Chest X-ray (CPU)** in Jupyter or VS Code. Open notebooks in numerical order.
The Next.js frontend uses a separate Node.js environment and communicates with FastAPI
over loopback. No container or cloud service is needed.

## Dataset

Source: [Chest X-Ray Images (Pneumonia), Paul Mooney / Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
The dataset originates from the pediatric chest X-ray research collection described by
[Kermany et al.](https://doi.org/10.1016/j.cell.2018.02.010).

The local root is `chest_xray/`, with `train`, `val`, and `test`, each containing `NORMAL`
and `PNEUMONIA`. A second extraction may exist at `chest_xray/chest_xray/`; the audit compares
it rather than loading both copies. For another location, edit `dataset_root` in
`configs/cpu.yaml`. Obtain and extract the dataset manually from Kaggle if it is absent.
The uploaded images are preserved in this private repository. Review the source dataset's
license and terms before reuse; the source code license does not cover images or pretrained weights.

```bash
python scripts/audit_dataset.py --root chest_xray
```

This decodes every image and records dimensions, byte hashes, decoded-pixel hashes, errors,
filename-derived patient proxies, duplicate copies, and cross-split overlap. The audit does
not silently discard corrupted images. Unknown groups or conflicting duplicate labels stop
manifest creation and require review.

The original test set is kept intact. One deterministic stratified group fold from the original
training pool becomes validation. The tiny original validation set is retained as `auxiliary`,
not used for model selection. Training-side images linked to test groups are excluded and
listed with reasons; development-side exact pixel duplicates are likewise listed. Filename
groups and hashes do **not** establish verified patient independence. Near duplicates and
unrecognized patient overlap can remain. Inspect the generated audit before interpreting results.

## CPU settings and preprocessing

Defaults in `configs/cpu.yaml`:

| Setting | Default | Reason |
| --- | --- | --- |
| Batch size | 8 | Conservative activation memory |
| Image size | 224 × 224 | Standard ResNet development resolution |
| Compute threads | 2 | Leave CPU capacity for the desktop |
| Loader workers | 0 | Avoid process memory and notebook worker overhead |
| Pinned memory | Disabled | CPU execution |
| Persistent workers | Disabled | Release resources between epochs |
| Precision | float32 | Predictable CPU behavior |
| Dataset cache | None | Decode batches on demand |

Images become RGB, are resized with bilinear interpolation and aspect-preserving black padding,
then normalized with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation
`[0.229, 0.224, 0.225]`. Training alone adds up to 5° rotation and 2% translation.
Evaluation and inference share the same deterministic transform. There are no flips or crops.

Reduce batch size to 4, then 2 if memory becomes tight. Lowering resolution to 160 is an explicit
experiment, not an automatic change; it is stored in the checkpoint and must remain consistent
for evaluation/inference. Do not change configuration midway through resume. Benchmark one or
two loader workers only after demonstrating an input-loading bottleneck. Avoid training and
web inference concurrently on this machine.

## Training

The default command is a capped **one-epoch smoke test** on 32 training and 16 validation
images, with no test-set access:

```bash
python -m ml.training.trainer
python -m ml.training.trainer --batch-size 4
```

Stage A freezes the backbone, including its BatchNorm statistics, and trains the one-logit
head. Binary cross-entropy operates on logits; sigmoid is used only for scores. Positive-class
loss weighting comes from training data; no oversampling sampler is combined with it.
Optional Stage B unfreezes only layer4 and the classifier at a lower learning rate. It is
disabled in the initial configuration.

After the owner approves full training:

```bash
python -m ml.training.trainer --full --approved-full-training
```

The flag records intentional execution; it is not an authentication mechanism. Do not invoke
it without the owner's approval. Initially there are at most five head epochs, early stopping
after three non-improving validation epochs, and checkpoint selection by validation ROC-AUC.
The threshold is fixed at 0.5; no clinical calibration or validation-based uncertainty policy
has been established.

Every epoch shows progress bars, timing, loss, and classification metrics. Checkpoints include
model weights, optimizer, scheduler, epoch boundary, random states, configuration, environment,
class mapping, transforms, threshold, and the manifest fingerprint. Ctrl+C or the notebook stop
button preserves the last complete boundary. An interrupted epoch is replayed from its start:

```bash
python -m ml.training.trainer --resume artifacts/checkpoints/smoke/last.pt
# Only after full-training approval:
python -m ml.training.trainer --full --approved-full-training --resume artifacts/checkpoints/run/last.pt
```

Checkpoints use atomic replacement and `torch.load(..., weights_only=True, map_location='cpu')`.
Use only trusted local checkpoints; restricted loading is not a guarantee against every
malformed or resource-exhausting artifact. Full training and smoke artifacts have separate
directories. Learning curves and history JSON are stored alongside each run.

## Evaluation, inference, and explanations

Notebook 05 requires an explicit `MODEL_SELECTION_LOCKED` switch. It refuses smoke artifacts,
checks the split fingerprint, and prevents overwriting an existing final evaluation. It writes
actual accuracy, sensitivity, specificity, precision, F1, ROC-AUC, class support, confusion
matrix, ROC and precision-recall curves, per-image scores, and classification reports under
`artifacts/reports/`. Undefined metrics are `null`; unsupported plots are omitted. Test results
must never guide further threshold, checkpoint, or hyperparameter selection.

Notebooks 06 and 07 use the shared `ml.inference.predictor.Predictor`. Grad-CAM differentiates
the selected class logit through ResNet18's layer4, weights activations by spatially averaged
gradients, applies ReLU, and normalizes the map. It works with frozen backbones by enabling
input gradients. A zero map stays zero. Overlays use model-input coordinates, including padding.
These maps describe model influence, not a verified lesion location.

Model scores are not calibrated probabilities of disease. False positives and false negatives,
class imbalance, uncertain patient independence, acquisition bias, pediatric population bias,
and domain shift restrict the conclusions possible from this dataset. This project does not
diagnose disease, replace radiologists, or recommend treatment.

## Verification

```bash
python -m pytest ml/tests -q
ruff check ml scripts
ruff format --check ml scripts
python scripts/execute_notebooks.py 01 02 03
python scripts/execute_notebooks.py 04 --smoke
python scripts/execute_notebooks.py 05 06 07
```

Executed notebooks are saved under ignored `artifacts/notebooks/`. Source notebooks contain no
outputs. The last command checks safe missing-model states without training or test evaluation.
Synthetic images/models are confined to automated tests; real experiments use the local dataset.

## Screenshots

Dashboard screenshots are pending the trained-model integration check. Executed notebooks
contain the dataset figures and training curves.

## Version control and next steps

Use the existing private [project repository](https://github.com/XxCPH4xX/Chest-X-ray-Pneumonia-Classification).
The uploaded dataset remains tracked. Model weights, reports, executed notebooks, databases,
and secrets stay local. See [repository notes](docs/repository.md).

Next: resume the saved training run, lock model selection, evaluate once on the held-out test
set, export the model, and check the app with real predictions.

Source code: MIT, see `LICENSE`. Dataset and pretrained weights retain their respective terms.
