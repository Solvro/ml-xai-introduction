from __future__ import annotations

import torch
from omegaconf import OmegaConf

from ml_xai_introduction.metrics.metric_base import MetricContext
from ml_xai_introduction.metrics.metric_factory import build_metric
from ml_xai_introduction.metrics.metric_manager import MetricsManager


def _context() -> MetricContext:
    predictions = torch.tensor([0, 1, 2, 0])
    targets = torch.tensor([0, 1, 1, 0])
    return MetricContext(predictions=predictions, targets=targets, loss_sum=1.2, num_samples=4)


def test_metric_registry_builds_accuracy() -> None:
    metric = build_metric(OmegaConf.create({}), "accuracy")
    assert metric.compute(_context()) == 0.75


def test_metrics_manager_filters_active_per_phase() -> None:
    cfg = OmegaConf.create({
        "train": {"active": ["loss"]},
        "validation": {"active": ["loss", "accuracy"]},
        "test": {"active": []},
    })
    manager = MetricsManager(cfg)
    context = _context()

    train_metrics = manager.compute("train", context)
    val_metrics = manager.compute("validation", context)
    test_metrics = manager.compute("test", context)

    assert set(train_metrics.keys()) == {"loss"}
    assert set(val_metrics.keys()) == {"loss", "accuracy"}
    assert test_metrics == {}
