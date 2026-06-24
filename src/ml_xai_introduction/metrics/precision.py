"""Precision metric plugin."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import precision_score

from ml_xai_introduction.metrics.metric_base import Metric, MetricContext
from ml_xai_introduction.metrics.metric_factory import metric_registry


class PrecisionMetric:
    def __init__(self, average: str) -> None:
        self._average = average

    def compute(self, context: MetricContext) -> float:
        predictions = context.predictions.cpu().numpy()
        targets = context.targets.cpu().numpy()
        return float(precision_score(targets, predictions, average=self._average, zero_division=0))


@metric_registry.register("precision")
def build_precision_metric(cfg: DictConfig) -> Metric:
    average = str(OmegaConf.select(cfg, "precision.average", default="macro"))
    return PrecisionMetric(average=average)
