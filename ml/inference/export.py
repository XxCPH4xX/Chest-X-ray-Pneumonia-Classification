"""Export a model with a matching test-set evaluation."""

import hashlib
import json
from pathlib import Path

from ml.training.checkpoint import read_checkpoint, save_checkpoint


def export_model(
    checkpoint, report="artifacts/reports/evaluation.json", destination="artifacts/model/model.pt"
):
    checkpoint = Path(checkpoint)
    metadata = read_checkpoint(checkpoint)
    if metadata.get("smoke", True):
        raise ValueError("Cannot export a smoke-test model")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    evaluation = json.loads(Path(report).read_text())
    if evaluation["checkpoint_sha256"] != digest:
        raise ValueError("Evaluation does not match selected checkpoint")
    if evaluation["threshold"] != metadata["threshold"]:
        raise ValueError("Evaluation threshold differs from selected checkpoint")
    keys = (
        "format_version",
        "architecture",
        "smoke",
        "class_mapping",
        "threshold",
        "preprocessing",
        "model_state",
        "config",
        "environment",
        "manifest_sha256",
    )
    artifact = {key: metadata[key] for key in keys}
    artifact.update(
        selection_locked=True,
        model_version=f"resnet18-{digest[:12]}",
        training_checkpoint_sha256=digest,
        uncertainty_policy=metadata.get("uncertainty_policy"),
    )
    save_checkpoint(artifact, destination)
    return artifact["model_version"]
