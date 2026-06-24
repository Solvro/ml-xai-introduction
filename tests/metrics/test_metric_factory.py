from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from ml_xai_introduction.metrics import metric_factory
from ml_xai_introduction.metrics.metric_factory import build_metric


def test_metric_registry_builds_loss() -> None:
    metric = build_metric(OmegaConf.create({}), "loss")
    assert metric.__class__.__name__ == "LossMetric"


def test_build_metric_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="Unknown metric"):
        build_metric(OmegaConf.create({}), "unknown")


def test_metric_registry_lists_registered_names() -> None:
    metric_factory._ensure_discovered()
    names = set(metric_factory.metric_registry._entries.keys())
    assert "loss" in names
    assert "accuracy" in names
    assert "f1" in names
