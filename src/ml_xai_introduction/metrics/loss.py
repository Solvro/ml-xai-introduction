"""Loss metric plugin."""

from __future__ import annotations

from omegaconf import DictConfig

from ml_xai_introduction.metrics.metric_base import Metric, MetricContext
from ml_xai_introduction.metrics.metric_factory import metric_registry


class LossMetric:
    def compute(self, context: MetricContext) -> float:
        return context.loss_sum / context.num_samples


@metric_registry.register("loss")
def build_loss_metric(cfg: DictConfig) -> Metric:
    del cfg
    return LossMetric()
