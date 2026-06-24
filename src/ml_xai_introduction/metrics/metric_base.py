"""Metric plugin contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
from omegaconf import DictConfig


@dataclass(frozen=True, slots=True)
class MetricContext:
    predictions: torch.Tensor
    targets: torch.Tensor
    loss_sum: float
    num_samples: int


@dataclass(slots=True)
class RunSummary:
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    best_val_accuracy: float = 0.0


class Metric(Protocol):
    def compute(self, context: MetricContext) -> float: ...


BuildMetricFn = Callable[[DictConfig], Metric]
