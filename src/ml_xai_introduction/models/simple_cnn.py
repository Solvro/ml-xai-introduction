"""Simple CNN model from Damian's branch (logits output for CrossEntropyLoss)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from omegaconf import DictConfig

from ml_xai_introduction.models.model_base import Model
from ml_xai_introduction.models.model_factory import model_registry


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 3, 1)
        self.conv2 = nn.Conv2d(5, 10, 3, 1, 2)
        self.conv3 = nn.Conv2d(10, 20, 3, 1, 2)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.normalize1 = nn.BatchNorm2d(5)
        self.normalize2 = nn.BatchNorm2d(10)
        self.normalize3 = nn.BatchNorm2d(20)
        self.drop2d = nn.Dropout2d(0.25)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(20 * 8 * 8, 200)
        self.fc2 = nn.Linear(200, 120)
        self.fc3 = nn.Linear(120, 60)
        self.fc4 = nn.Linear(60, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.normalize1(x)
        x = F.relu(self.conv2(x))
        x = self.normalize2(x)
        x = self.pool1(x)
        x = self.drop2d(x)
        x = F.relu(self.conv3(x))
        x = self.normalize3(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        return self.fc4(x)


@model_registry.register("simple_cnn")
def build_simple_cnn(cfg: DictConfig) -> Model:
    return SimpleCNN(int(cfg.data.num_classes))
