import time

import torch
from PIL import Image

from ml.data.transforms import Letterbox, preprocessing, specification
from ml.explainability.gradcam import gradcam, overlay
from ml.runtime import configure
from ml.training.checkpoint import load_model


class Predictor:
    def __init__(self, checkpoint, allow_smoke=False):
        configure()
        self.model, self.metadata = load_model(checkpoint, allow_smoke)
        self.size = self.metadata["preprocessing"]["image_size"]
        if self.metadata["preprocessing"] != specification(self.size):
            raise ValueError("Checkpoint preprocessing does not match this implementation")
        self.transform = preprocessing(self.size)

    def predict(self, image: Image.Image, explain=False):
        started = time.perf_counter()
        tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            score = float(self.model(tensor).flatten()[0].sigmoid())
        threshold = self.metadata["threshold"]
        target = int(score >= threshold)
        policy = self.metadata.get("uncertainty_policy")
        inconclusive = bool(policy and policy["low"] <= score <= policy["high"])
        result = {
            "predicted_class": "INCONCLUSIVE"
            if inconclusive
            else ("PNEUMONIA" if target else "NORMAL"),
            "probabilities": {"NORMAL": 1 - score, "PNEUMONIA": score},
            "threshold": threshold,
            "architecture": "resnet18",
            "smoke": self.metadata["smoke"],
            "gradcam_available": explain,
            "gradcam_target": "PNEUMONIA" if target else "NORMAL",
            "uncertainty_policy": policy,
        }
        heatmap, combined = None, None
        if explain:
            heatmap = gradcam(self.model, tensor, target)
            combined = overlay(Letterbox(self.size)(image), heatmap)
        result["seconds"] = time.perf_counter() - started
        return result, heatmap, combined
