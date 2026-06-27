"""Loss factory backed by a config-driven registry."""

from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig

from pytorch_research_template.losses.loss_base import BuildLossFn
from pytorch_research_template.registry import Registry, autodiscover, normalize_name

loss_registry: Registry[BuildLossFn] = Registry("loss")

_DISCOVERED = False


def _ensure_discovered() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    autodiscover("pytorch_research_template.losses", frozenset({"loss_factory", "loss_base"}))
    _DISCOVERED = True


def load_loss(cfg: DictConfig, num_classes: int) -> nn.Module:
    """Build the loss function selected in cfg."""

    _ensure_discovered()
    build_fn = loss_registry.get(normalize_name(cfg.name))
    return build_fn(cfg, num_classes)
