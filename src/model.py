from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torchvision import models


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.conv2(x)
        return self.act(x + residual)


class DogHeartCNN(nn.Module):
    """A compact custom CNN for dog chest X-ray heart-size classification."""

    def __init__(self, num_classes: int = 3, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(3, 32, stride=2),
            ConvBNAct(32, 48),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResidualBlock(48, 64, stride=1, dropout=0.05),
            ResidualBlock(64, 96, stride=2, dropout=0.08),
            ResidualBlock(96, 128, stride=1, dropout=0.08),
            ResidualBlock(128, 192, stride=2, dropout=0.10),
            ResidualBlock(192, 256, stride=1, dropout=0.10),
            ResidualBlock(256, 320, stride=2, dropout=0.12),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(320, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


@dataclass(frozen=True)
class TorchvisionModelSpec:
    builder: Callable[..., nn.Module]
    default_weights: Any
    replace_head: Callable[[nn.Module, int, float], None]


def _dropout_classifier(in_features: int, num_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )


def _replace_fc_with_dropout(model: nn.Module, num_classes: int, dropout: float) -> None:
    in_features = model.fc.in_features
    model.fc = _dropout_classifier(in_features, num_classes, dropout)


def _replace_fc(model: nn.Module, num_classes: int, dropout: float) -> None:
    del dropout
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)


def _replace_classifier_with_dropout(model: nn.Module, num_classes: int, dropout: float) -> None:
    in_features = model.classifier.in_features
    model.classifier = _dropout_classifier(in_features, num_classes, dropout)


def _replace_classifier_tail_with_dropout(model: nn.Module, num_classes: int, dropout: float) -> None:
    in_features = model.classifier[-1].in_features
    model.classifier = _dropout_classifier(in_features, num_classes, dropout)


def _replace_classifier_tail(model: nn.Module, num_classes: int, dropout: float) -> None:
    del dropout
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)


def _replace_head(model: nn.Module, num_classes: int, dropout: float) -> None:
    del dropout
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)


TORCHVISION_MODEL_SPECS: dict[str, TorchvisionModelSpec] = {
    "resnet18": TorchvisionModelSpec(models.resnet18, models.ResNet18_Weights.DEFAULT, _replace_fc_with_dropout),
    "resnet34": TorchvisionModelSpec(models.resnet34, models.ResNet34_Weights.DEFAULT, _replace_fc_with_dropout),
    "resnet50": TorchvisionModelSpec(models.resnet50, models.ResNet50_Weights.DEFAULT, _replace_fc_with_dropout),
    "densenet121": TorchvisionModelSpec(
        models.densenet121,
        models.DenseNet121_Weights.DEFAULT,
        _replace_classifier_with_dropout,
    ),
    "efficientnet_b0": TorchvisionModelSpec(
        models.efficientnet_b0,
        models.EfficientNet_B0_Weights.DEFAULT,
        _replace_classifier_tail_with_dropout,
    ),
    "efficientnet_b2": TorchvisionModelSpec(
        models.efficientnet_b2,
        models.EfficientNet_B2_Weights.DEFAULT,
        _replace_classifier_tail_with_dropout,
    ),
    "efficientnet_v2_s": TorchvisionModelSpec(
        models.efficientnet_v2_s,
        models.EfficientNet_V2_S_Weights.DEFAULT,
        _replace_classifier_tail_with_dropout,
    ),
    "convnext_tiny": TorchvisionModelSpec(
        models.convnext_tiny,
        models.ConvNeXt_Tiny_Weights.DEFAULT,
        _replace_classifier_tail,
    ),
    "convnext_small": TorchvisionModelSpec(
        models.convnext_small,
        models.ConvNeXt_Small_Weights.DEFAULT,
        _replace_classifier_tail,
    ),
    "mobilenet_v3_large": TorchvisionModelSpec(
        models.mobilenet_v3_large,
        models.MobileNet_V3_Large_Weights.DEFAULT,
        _replace_classifier_tail,
    ),
    "regnet_y_400mf": TorchvisionModelSpec(
        models.regnet_y_400mf,
        models.RegNet_Y_400MF_Weights.DEFAULT,
        _replace_fc,
    ),
    "regnet_y_800mf": TorchvisionModelSpec(
        models.regnet_y_800mf,
        models.RegNet_Y_800MF_Weights.DEFAULT,
        _replace_fc,
    ),
    "swin_t": TorchvisionModelSpec(models.swin_t, models.Swin_T_Weights.DEFAULT, _replace_head),
}

SUPPORTED_ARCHITECTURES: tuple[str, ...] = ("custom", *TORCHVISION_MODEL_SPECS.keys())


def _resolve_weights(default_weights: Any, pretrained: bool) -> Any:
    return default_weights if pretrained else None


def build_model(
    num_classes: int = 3,
    dropout: float = 0.25,
    architecture: str = "custom",
    pretrained: bool = False,
) -> nn.Module:
    architecture = architecture.lower()
    if architecture == "custom":
        return DogHeartCNN(num_classes=num_classes, dropout=dropout)

    spec = TORCHVISION_MODEL_SPECS.get(architecture)
    if spec is None:
        supported = ", ".join(SUPPORTED_ARCHITECTURES)
        raise ValueError(f"Unknown architecture: {architecture}. Supported architectures: {supported}")

    model = spec.builder(weights=_resolve_weights(spec.default_weights, pretrained))
    spec.replace_head(model, num_classes, dropout)
    return model
