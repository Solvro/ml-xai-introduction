"""Metric plugins and registry factory."""

from ml_xai_introduction.metrics.metric_base import BuildMetricFn, Metric, MetricContext, RunSummary
from ml_xai_introduction.metrics.metric_factory import metric_registry
from ml_xai_introduction.metrics.metric_logger import MetricLogger
from ml_xai_introduction.metrics.metric_manager import MetricsManager

__all__ = [
    "BuildMetricFn",
    "Metric",
    "MetricContext",
    "MetricLogger",
    "MetricsManager",
    "RunSummary",
    "metric_registry",
]
