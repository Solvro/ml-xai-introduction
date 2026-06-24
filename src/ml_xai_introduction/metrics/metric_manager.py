"""Config-driven metric computation."""

from __future__ import annotations

from omegaconf import DictConfig

from ml_xai_introduction.metrics.metric_base import MetricContext
from ml_xai_introduction.metrics.metric_factory import build_metric
from ml_xai_introduction.registry import normalize_name


class MetricsManager:
    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg

    def active_for(self, phase: str) -> list[str]:
        phase_cfg = self._cfg.get(phase)
        if phase_cfg is None:
            return []
        active = phase_cfg.get("active", [])
        return [normalize_name(name) for name in active]

    def compute(self, phase: str, context: MetricContext) -> dict[str, float]:
        results: dict[str, float] = {}
        for name in self.active_for(phase):
            metric = build_metric(self._cfg, name)
            results[name] = metric.compute(context)
        return results
