"""Model factory backed by a config-driven registry."""

from __future__ import annotations

from omegaconf import DictConfig

from ml_xai_introduction.models.model_base import BuildModelFn, Model
from ml_xai_introduction.registry import Registry, autodiscover, normalize_name

model_registry: Registry[BuildModelFn] = Registry("model")

_DISCOVERED = False


def _ensure_discovered() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    autodiscover("ml_xai_introduction.models", frozenset({"model_factory", "model_base"}))
    _DISCOVERED = True


def load_model(cfg: DictConfig) -> Model:
    """Build the model selected in cfg."""

    _ensure_discovered()
    build_fn = model_registry.get(normalize_name(cfg.model.name))
    return build_fn(cfg)
