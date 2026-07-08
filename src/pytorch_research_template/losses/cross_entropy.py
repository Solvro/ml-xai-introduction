"""Cross-entropy loss plugin."""

from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig

from pytorch_research_template.losses.loss_factory import loss_registry


@loss_registry.register("cross_entropy", "ce")
def build_cross_entropy(cfg: DictConfig, num_classes: int) -> nn.Module:
    del num_classes
    return nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.0)))
