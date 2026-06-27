"""Accuracy metric plugin."""

from __future__ import annotations

from omegaconf import DictConfig

from pytorch_research_template.metrics.metric_base import Metric, MetricContext
from pytorch_research_template.metrics.metric_factory import metric_registry


class AccuracyMetric:
    def compute(self, context: MetricContext) -> float:
        correct = int((context.predictions == context.targets).sum().item())
        return correct / context.num_samples


@metric_registry.register("accuracy")
def build_accuracy_metric(cfg: DictConfig) -> Metric:
    del cfg
    return AccuracyMetric()
