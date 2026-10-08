import hashlib
import json

import pytest

from ml.data.dataset import CLASSES
from ml.data.transforms import specification
from ml.inference.export import export_model
from ml.models.resnet import build_model
from ml.training.checkpoint import load_model, read_checkpoint, save_checkpoint


def test_export_requires_matching_evaluation_and_strips_training_state(tmp_path):
    checkpoint = tmp_path / "best.pt"
    metadata = {
        "format_version": 1,
        "architecture": "resnet18",
        "smoke": False,
        "class_mapping": CLASSES,
        "threshold": 0.5,
        "preprocessing": specification(64),
        "model_state": build_model(False).state_dict(),
        "config": {"device": "cpu"},
        "environment": {},
        "manifest_sha256": "test-manifest",
        "optimizer_state": {},
    }
    save_checkpoint(metadata, checkpoint)
    report = tmp_path / "evaluation.json"
    report.write_text(json.dumps({"checkpoint_sha256": "other", "threshold": 0.5}))
    with pytest.raises(ValueError, match="does not match"):
        export_model(checkpoint, report, tmp_path / "model.pt")
    report.write_text(
        json.dumps(
            {
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "threshold": 0.5,
            }
        )
    )
    destination = tmp_path / "model.pt"
    version = export_model(checkpoint, report, destination)
    _, exported = load_model(destination)
    assert exported["selection_locked"] is True
    assert exported["model_version"] == version
    assert "optimizer_state" not in exported


def test_uncertainty_policy_rejects_non_validation_source(tmp_path):
    path = tmp_path / "invalid.pt"
    save_checkpoint(
        {
            "format_version": 1,
            "architecture": "resnet18",
            "class_mapping": CLASSES,
            "threshold": 0.5,
            "uncertainty_policy": {"low": 0.4, "high": 0.6, "source": "test"},
        },
        path,
    )
    with pytest.raises(ValueError, match="validation"):
        read_checkpoint(path)
