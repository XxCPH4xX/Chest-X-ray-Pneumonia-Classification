from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(pretrained: bool = True):
    """Create a one-logit ResNet18; downloads ImageNet weights only when requested."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model.to("cpu")


def set_stage(model, stage: str):
    if stage not in ("head", "finetune"):
        raise ValueError(f"Unknown stage: {stage}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            name.startswith("fc.") or (stage == "finetune" and name.startswith("layer4."))
        )


def training_mode(model, stage: str):
    # Keep frozen BatchNorm statistics fixed during feature extraction.
    model.eval()
    model.fc.train()
    if stage == "finetune":
        model.layer4.train()
