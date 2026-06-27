"""Baseline fully connected model."""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from pytorch_research_template.models.model_base import Model
from pytorch_research_template.models.model_factory import model_registry


class BaselineNN(nn.Module):
    """Flatten → FC(auto→128) → FC(128→num_classes)."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


@model_registry.register("baseline")
def build_baseline(cfg: DictConfig) -> Model:
    return BaselineNN(int(cfg.data.num_classes))
