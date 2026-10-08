import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


def gradcam(model, image: torch.Tensor, target: int = 1) -> np.ndarray:
    """Explain the positive logit (1) or its negative (0), including frozen backbones."""
    if target not in (0, 1):
        raise ValueError("Target must be NORMAL=0 or PNEUMONIA=1")
    activation = []
    handle = model.layer4.register_forward_hook(lambda _m, _i, output: activation.append(output))
    was_training = model.training
    try:
        model.eval()
        with torch.enable_grad():
            image = image.detach().clone().to("cpu").requires_grad_(True)
            logit = model(image).flatten()[0]
            gradient = torch.autograd.grad(logit if target else -logit, activation[0])[0]
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activation[0]).sum(dim=1, keepdim=True).relu()
            cam = F.interpolate(cam, image.shape[-2:], mode="bilinear", align_corners=False)
            cam = cam[0, 0].detach().numpy()
            maximum = cam.max()
            return cam / maximum if maximum > 0 else np.zeros_like(cam)
    finally:
        handle.remove()
        model.train(was_training)


def overlay(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.35) -> Image.Image:
    """Overlay on the same letterboxed input coordinates used by the model."""
    from matplotlib import colormaps

    heat = Image.fromarray((colormaps["inferno"](heatmap)[..., :3] * 255).astype("uint8"))
    return Image.blend(image.convert("RGB"), heat.resize(image.size), alpha)
