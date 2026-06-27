"""Model factory backed by a config-driven registry."""

from __future__ import annotations

from omegaconf import DictConfig

from pytorch_research_template.models.model_base import BuildModelFn, Model
from pytorch_research_template.registry import Registry, autodiscover, normalize_name

model_registry: Registry[BuildModelFn] = Registry("model")

_DISCOVERED = False


def _ensure_discovered() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    autodiscover("pytorch_research_template.models", frozenset({"model_factory", "model_base"}))
    _DISCOVERED = True


def load_model(cfg: DictConfig) -> Model:
    """Build the model selected in cfg."""

    _ensure_discovered()
    build_fn = model_registry.get(normalize_name(cfg.model.name))
    return build_fn(cfg)
