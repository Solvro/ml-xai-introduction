"""Recall metric plugin."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import recall_score

from ml_xai_introduction.metrics.metric_base import Metric, MetricContext
from ml_xai_introduction.metrics.metric_factory import metric_registry


class RecallMetric:
    def __init__(self, average: str) -> None:
        self._average = average

    def compute(self, context: MetricContext) -> float:
        predictions = context.predictions.cpu().numpy()
        targets = context.targets.cpu().numpy()
        return float(recall_score(targets, predictions, average=self._average, zero_division=0))


@metric_registry.register("recall")
def build_recall_metric(cfg: DictConfig) -> Metric:
    average = str(OmegaConf.select(cfg, "recall.average", default="macro"))
    return RecallMetric(average=average)
