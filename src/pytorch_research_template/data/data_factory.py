"""Dataset factory backed by a config-driven registry."""

from __future__ import annotations

from omegaconf import DictConfig

from pytorch_research_template.data.data_base import DataBundle, LoadDataFn
from pytorch_research_template.registry import Registry, autodiscover, normalize_name

dataset_registry: Registry[LoadDataFn] = Registry("dataset")

_DISCOVERED = False


def _ensure_discovered() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    autodiscover("pytorch_research_template.data", frozenset({"data_factory", "data_base"}))
    _DISCOVERED = True


def load_data(cfg: DictConfig) -> DataBundle:
    """Return train, validation, and test dataloaders for the dataset selected in cfg."""

    _ensure_discovered()
    build_fn = dataset_registry.get(normalize_name(cfg.data.name))
    return build_fn(cfg)
