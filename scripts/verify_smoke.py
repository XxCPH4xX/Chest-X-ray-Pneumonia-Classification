"""Verify real smoke checkpoint reload and explanations without presenting a trained model."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from ml.data.transforms import Letterbox
from ml.inference.predictor import Predictor
from ml.training.checkpoint import load_model

config = yaml.safe_load(Path("configs/cpu.yaml").read_text())
output = Path(config["checkpoint_dir"]) / "smoke"
path = output / "best.pt"
predictor = Predictor(path, allow_smoke=True)
manifest = pd.read_csv(config["manifest"])
row = manifest[manifest.split.eq("val")].iloc[0]
with Image.open(Path(config["dataset_root"]) / row.path) as source:
    image = source.convert("RGB")
result, heatmap, combined = predictor.predict(image, explain=True)
assert result["smoke"]
assert np.isfinite(heatmap).all()
assert heatmap.shape == (config["image_size"], config["image_size"])
assert abs(sum(result["probabilities"].values()) - 1) < 1e-6
reloaded, _ = load_model(path, allow_smoke=True)
with torch.no_grad():
    sample = predictor.transform(image).unsqueeze(0)
    assert torch.equal(predictor.model(sample), reloaded(sample))
try:
    Predictor(path)
except ValueError:
    blocked = True
else:
    raise AssertionError("Production predictor unexpectedly accepted a smoke checkpoint")
Letterbox(config["image_size"])(image).save(output / "gradcam_input.png")
combined.save(output / "gradcam_overlay.png")
Image.fromarray((heatmap * 255).astype("uint8")).save(output / "gradcam_heatmap.png")
summary = {
    "smoke_only": True,
    "checkpoint_reload_exact": True,
    "normal_inference_rejects_smoke": blocked,
    "real_image_gradcam_finite": True,
    "gradcam_min": float(heatmap.min()),
    "gradcam_max": float(heatmap.max()),
    "inference_with_gradcam_seconds": result["seconds"],
    "source_split": "val",
}
(output / "verification.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
