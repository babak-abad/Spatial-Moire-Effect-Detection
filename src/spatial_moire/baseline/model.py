from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from config import CLASSIFIER_OUT_FEATURES


def build_resnet18_classifier(pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=CLASSIFIER_OUT_FEATURES)
    return model


def load_model_checkpoint(model: nn.Module, checkpoint_path: Path) -> dict:
    checkpoint = torch.load(f=checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict=checkpoint["model_state"])
    return checkpoint
