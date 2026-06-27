"""Metric plugins and registry factory."""

from pytorch_research_template.metrics.metric_base import BuildMetricFn, Metric, MetricContext, RunSummary
from pytorch_research_template.metrics.metric_factory import metric_registry
from pytorch_research_template.metrics.metric_logger import MetricLogger
from pytorch_research_template.metrics.metric_manager import MetricsManager

__all__ = [
    "BuildMetricFn",
    "Metric",
    "MetricContext",
    "MetricLogger",
    "MetricsManager",
    "RunSummary",
    "metric_registry",
]
