"""Preserve the complete image with aspect-ratio letterboxing and ImageNet normalization."""

from PIL import Image, ImageOps
from torchvision import transforms as T

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class Letterbox:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, image):
        return ImageOps.pad(
            image.convert("RGB"), (self.size, self.size), method=Image.Resampling.BILINEAR, color=0
        )


def preprocessing(size: int = 224, training: bool = False):
    if not 64 <= size <= 512:
        raise ValueError("Image size must be between 64 and 512")
    ops = [Letterbox(size)]
    if training:
        ops.append(T.RandomAffine(degrees=5, translate=(0.02, 0.02), fill=0))
    return T.Compose([*ops, T.ToTensor(), T.Normalize(MEAN, STD)])


def specification(size: int):
    return {
        "image_size": size,
        "resize": "letterbox",
        "interpolation": "bilinear",
        "color": "RGB",
        "mean": MEAN,
        "std": STD,
    }
