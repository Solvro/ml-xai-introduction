"""Metric factory backed by a config-driven registry."""

from __future__ import annotations

from omegaconf import DictConfig

from ml_xai_introduction.metrics.metric_base import BuildMetricFn, Metric
from ml_xai_introduction.registry import Registry, autodiscover, normalize_name

metric_registry: Registry[BuildMetricFn] = Registry("metric")

_DISCOVERED = False


def _ensure_discovered() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    autodiscover(
        "ml_xai_introduction.metrics",
        frozenset({"metric_factory", "metric_base", "metric_manager", "metric_logger"}),
    )
    _DISCOVERED = True


def build_metric(cfg: DictConfig, name: str) -> Metric:
    _ensure_discovered()
    build_fn = metric_registry.get(normalize_name(name))
    return build_fn(cfg)
