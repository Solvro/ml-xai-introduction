"""Convolutional model for grayscale image classification."""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ml_xai_introduction.models.model_base import Model
from ml_xai_introduction.models.model_factory import model_registry


class CNN(nn.Module):
    """Conv(1→8) → Conv(8→16) → MaxPool → Conv(16→24) → MaxPool → FC(auto→100) → FC(100→num_classes)."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(100)
        self.output = nn.Linear(100, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.fc1(self.flatten(x)))
        return self.output(x)


@model_registry.register("cnn")
def build_cnn(cfg: DictConfig) -> Model:
    return CNN(int(cfg.data.num_classes))
