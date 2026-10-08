import os
from pathlib import Path

import torch

from ml.models.resnet import build_model


def save_checkpoint(payload: dict, path: str | Path):
    """Atomically replace checkpoints so interruption cannot leave a partial target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def read_checkpoint(path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("architecture") != "resnet18":
        raise ValueError("Unsupported checkpoint format or architecture")
    if checkpoint.get("class_mapping") != {"NORMAL": 0, "PNEUMONIA": 1}:
        raise ValueError("Invalid class mapping")
    if not 0 < checkpoint["threshold"] < 1:
        raise ValueError("Threshold must be between 0 and 1")
    policy = checkpoint.get("uncertainty_policy")
    if policy and not (
        policy.get("source") == "validation"
        and 0 <= policy["low"] <= checkpoint["threshold"] <= policy["high"] <= 1
    ):
        raise ValueError(
            "Uncertainty policy must be derived from validation and span the threshold"
        )
    return checkpoint


def load_model(path: str | Path, allow_smoke: bool = False):
    checkpoint = read_checkpoint(path)
    if checkpoint.get("smoke", True) and not allow_smoke:
        raise ValueError("Smoke-test checkpoints cannot be used as a research inference model")
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint
