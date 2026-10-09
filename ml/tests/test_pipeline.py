from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from ml.data.dataset import CLASSES, XRayDataset, audit_dataset, build_manifest
from ml.data.transforms import preprocessing, specification
from ml.evaluation.metrics import classification_metrics
from ml.explainability.gradcam import gradcam
from ml.inference.predictor import Predictor
from ml.models.resnet import build_model, set_stage
from ml.runtime import configure
from ml.training.checkpoint import load_model, save_checkpoint
from ml.training.trainer import restore_rng, rng_state, train


@pytest.fixture(autouse=True)
def cpu_runtime():
    configure(42, 2)


@pytest.fixture
def small_dataset(tmp_path):
    root = tmp_path / "dataset"
    rng = np.random.default_rng(42)
    for split, start in [("train", 1), ("val", 100), ("test", 200)]:
        for label in CLASSES:
            directory = root / split / label
            directory.mkdir(parents=True)
            for i in range(start, start + 12):
                name = (
                    f"person{i}_bacteria_1.png" if label == "PNEUMONIA" else f"IM-{i:04}-0001.png"
                )
                Image.fromarray(rng.integers(0, 256, (70, 80), dtype=np.uint8)).save(
                    directory / name
                )
    return root


def test_audit_split_and_loading(small_dataset, tmp_path):
    inventory = audit_dataset(small_dataset, tmp_path / "audit")
    manifest = build_manifest(inventory, tmp_path / "manifest.csv")
    other = build_manifest(inventory, tmp_path / "other.csv")
    pd.testing.assert_frame_equal(manifest, other)
    assert set(manifest[manifest.source_split == "test"].split) == {"test"}
    assert not (
        set(manifest[manifest.split == "train"].group)
        & set(manifest[manifest.split == "val"].group)
    )
    image, target = XRayDataset(small_dataset, manifest, "train", preprocessing(64))[0]
    assert image.shape == (3, 64, 64)
    assert target in (0, 1)
    assert torch.isfinite(image).all()


def test_corruption_is_reported(small_dataset, tmp_path):
    (small_dataset / "train/NORMAL/IM-9999-0001.png").write_bytes(b"bad")
    inventory = audit_dataset(small_dataset, tmp_path / "audit")
    assert inventory.error.ne("").sum() == 1
    with pytest.raises(ValueError, match="Unreadable"):
        build_manifest(inventory, tmp_path / "manifest.csv")


def test_overlap_excluded_and_test_preserved(small_dataset, tmp_path):
    source = small_dataset / "test/PNEUMONIA/person200_bacteria_1.png"
    (small_dataset / "train/PNEUMONIA/person200_bacteria_2.png").write_bytes(source.read_bytes())
    manifest = build_manifest(audit_dataset(small_dataset, tmp_path / "audit"), tmp_path / "m.csv")
    row = manifest[manifest.path.str.endswith("person200_bacteria_2.png")].iloc[0]
    assert row.split == "excluded"
    assert row.exclusion_reason == "group_overlaps_test"
    assert sum(manifest.split == "test") == 24


def test_metrics_undefined():
    m = classification_metrics([0, 0], [0.1, 0.2])
    assert m["specificity"] == 1
    assert m["sensitivity"] is None
    assert m["precision"] is None
    assert m["roc_auc"] is None
    assert classification_metrics([0, 1], [0.2, 0.8])["confusion_matrix"] == [[1, 0], [0, 1]]


def test_resnet_gradcam_and_checkpoint(tmp_path):
    model = build_model(False)
    set_stage(model, "head")
    model.eval()
    image = torch.rand(1, 3, 64, 64)
    assert model(image).shape == (1, 1)
    heat = gradcam(model, image)
    assert heat.shape == (64, 64)
    assert np.isfinite(heat).all()
    assert 0 <= heat.min() <= heat.max() <= 1
    path = tmp_path / "model.pt"
    save_checkpoint(
        {
            "format_version": 1,
            "architecture": "resnet18",
            "smoke": True,
            "class_mapping": CLASSES,
            "threshold": 0.5,
            "preprocessing": specification(64),
            "model_state": model.state_dict(),
        },
        path,
    )
    with pytest.raises(ValueError, match="Smoke"):
        load_model(path)
    predictor = Predictor(path, allow_smoke=True)
    result, _, _ = predictor.predict(Image.new("RGB", (80, 70)))
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    assert result["smoke"]


def test_full_training_guard():
    with pytest.raises(PermissionError):
        train({}, smoke=False)


def test_rng_roundtrip():
    generator = torch.Generator().manual_seed(4)
    state = rng_state(generator)
    expected = torch.rand(8), torch.rand(8, generator=generator), np.random.rand(8)
    restore_rng(state, generator)
    assert torch.equal(torch.rand(8), expected[0])
    assert torch.equal(torch.rand(8, generator=generator), expected[1])
    np.testing.assert_array_equal(np.random.rand(8), expected[2])


def test_training_resume_boundary(small_dataset, tmp_path, monkeypatch):
    import ml.training.trainer as trainer

    manifest = tmp_path / "manifest.csv"
    build_manifest(audit_dataset(small_dataset, tmp_path / "audit"), manifest)
    config = {
        "device": "cpu",
        "seed": 42,
        "threads": 2,
        "manifest": str(manifest),
        "dataset_root": str(small_dataset),
        "image_size": 64,
        "batch_size": 8,
        "num_workers": 0,
        "pretrained": False,
        "head_epochs": 1,
        "fine_tune_epochs": 0,
        "learning_rate_head": 0.001,
        "learning_rate_finetune": 0.00001,
        "weight_decay": 0.0001,
        "threshold": 0.5,
        "early_stopping_patience": 3,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
    }
    original = trainer.run_epoch

    def interrupt(*args, **kwargs):
        original(*args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(trainer, "run_epoch", interrupt)
    assert train(config) == []
    path = Path(config["checkpoint_dir"]) / "smoke/last.pt"
    assert path.exists()
    monkeypatch.setattr(trainer, "run_epoch", original)
    resumed = train(config, resume=str(path))
    assert len(resumed) == 1
    assert resumed[0]["val"]["roc_auc"] is not None
    # A completed smoke run must not silently add more epochs on resume.
    assert train(config, resume=str(path)) == resumed
    config["checkpoint_dir"] = str(tmp_path / "fresh")
    fresh = train(config)
    recovered_weights = torch.load(path, weights_only=True)["model_state"]
    fresh_weights = torch.load(tmp_path / "fresh/smoke/last.pt", weights_only=True)["model_state"]
    assert all(torch.equal(recovered_weights[k], fresh_weights[k]) for k in fresh_weights)
    assert fresh[0]["val"]["loss"] == resumed[0]["val"]["loss"]
