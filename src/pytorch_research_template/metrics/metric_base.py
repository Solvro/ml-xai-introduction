"""Metric plugin contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, cast

import torch
from omegaconf import DictConfig

SklearnAverage = Literal["binary", "micro", "macro", "weighted", "samples"]
ALLOWED_SKLEARN_AVERAGES = frozenset({"binary", "micro", "macro", "weighted", "samples"})


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


def parse_sklearn_average(raw: object) -> SklearnAverage:
    average = str(raw).strip().lower()
    if average not in ALLOWED_SKLEARN_AVERAGES:
        allowed = ", ".join(sorted(ALLOWED_SKLEARN_AVERAGES))
        raise ValueError(f"Invalid average {raw!r}; expected one of: {allowed}")
    return cast(SklearnAverage, average)
