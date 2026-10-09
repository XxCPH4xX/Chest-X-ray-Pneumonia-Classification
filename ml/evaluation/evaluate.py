"""One-time held-out evaluation after validation-based model selection is locked."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
)
from tqdm.auto import tqdm

from ml.data.dataset import XRayDataset
from ml.data.transforms import preprocessing
from ml.evaluation.metrics import classification_metrics
from ml.runtime import configure
from ml.training.checkpoint import load_model
from ml.training.trainer import make_loader


def evaluate(checkpoint, config, output="artifacts/reports", selection_locked=False):
    """Refuse smoke checkpoints and repeated test evaluation in the same report directory."""
    if not selection_locked:
        raise PermissionError("Lock model selection and threshold using validation data first")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "evaluation.json").exists():
        raise FileExistsError("Evaluation already exists; do not repeatedly tune against test data")
    configure(config["seed"], config["threads"])
    model, metadata = load_model(checkpoint)
    manifest_path = Path(config["manifest"])
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != metadata["manifest_sha256"]:
        raise ValueError("Evaluation manifest differs from training")
    dataset = XRayDataset(
        config["dataset_root"],
        pd.read_csv(manifest_path),
        "test",
        preprocessing(metadata["preprocessing"]["image_size"]),
    )
    labels, scores = [], []
    with torch.no_grad():
        for images, targets in tqdm(make_loader(dataset, config), desc="Held-out evaluation"):
            scores.extend(model(images).flatten().sigmoid().tolist())
            labels.extend(targets.tolist())
    metrics = classification_metrics(labels, scores, metadata["threshold"])
    predictions = [int(x >= metadata["threshold"]) for x in scores]
    report = classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["NORMAL", "PNEUMONIA"],
        output_dict=True,
        zero_division=float("nan"),
    )
    # JSON has no NaN; retain undefined metrics explicitly as null.
    import math

    report = {
        k: {a: None if isinstance(b, float) and math.isnan(b) else b for a, b in v.items()}
        if isinstance(v, dict)
        else v
        for k, v in report.items()
    }
    details = dataset.rows[["path", "target"]].copy()
    details["score"] = scores
    details["prediction"] = predictions
    details.to_csv(output / "predictions.csv", index=False)
    for filename, plot in [
        (
            "confusion_matrix.png",
            lambda: ConfusionMatrixDisplay.from_predictions(
                labels, predictions, labels=[0, 1], display_labels=["NORMAL", "PNEUMONIA"]
            ),
        ),
        ("roc_curve.png", lambda: RocCurveDisplay.from_predictions(labels, scores)),
        (
            "precision_recall_curve.png",
            lambda: PrecisionRecallDisplay.from_predictions(labels, scores),
        ),
    ]:
        if filename != "confusion_matrix.png" and len(set(labels)) < 2:
            continue
        display = plot()
        display.figure_.tight_layout()
        display.figure_.savefig(output / filename)
        plt.close(display.figure_)
    (output / "classification_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    metrics.update(
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        manifest_sha256=metadata["manifest_sha256"],
        threshold=metadata["threshold"],
    )
    (output / "evaluation.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    return metrics
