from __future__ import annotations

import pytest
import torch.nn as nn
from omegaconf import OmegaConf

from pytorch_research_template.losses import loss_factory


def test_load_loss_dispatches_to_registered_builder(monkeypatch) -> None:
    calls: list[tuple[object, int]] = []

    def fake_builder(cfg: object, num_classes: int) -> nn.Module:
        calls.append((cfg, num_classes))
        return nn.CrossEntropyLoss()

    monkeypatch.setattr(loss_factory, "_DISCOVERED", True)
    monkeypatch.setitem(loss_factory.loss_registry._entries, "cross_entropy", fake_builder)

    cfg = OmegaConf.create({"name": "cross_entropy"})
    loss = loss_factory.load_loss(cfg, num_classes=10)

    assert isinstance(loss, nn.CrossEntropyLoss)
    assert calls == [(cfg, 10)]


def test_load_loss_rejects_unknown_loss() -> None:
    cfg = OmegaConf.create({"name": "unknown"})

    with pytest.raises(ValueError, match="Unknown loss"):
        loss_factory.load_loss(cfg, num_classes=10)
