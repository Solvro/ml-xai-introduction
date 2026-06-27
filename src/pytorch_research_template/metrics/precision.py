"""Precision metric plugin."""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import precision_score

from pytorch_research_template.metrics.metric_base import Metric, MetricContext, SklearnAverage, parse_sklearn_average
from pytorch_research_template.metrics.metric_factory import metric_registry


class PrecisionMetric:
    def __init__(self, average: SklearnAverage) -> None:
        self._average = average

    def compute(self, context: MetricContext) -> float:
        predictions = context.predictions.cpu().numpy()
        targets = context.targets.cpu().numpy()
        return float(precision_score(targets, predictions, average=self._average, zero_division=0))


@metric_registry.register("precision")
def build_precision_metric(cfg: DictConfig) -> Metric:
    average = parse_sklearn_average(OmegaConf.select(cfg, "precision.average", default="macro"))
    return PrecisionMetric(average=average)
