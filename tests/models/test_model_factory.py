from __future__ import annotations

import pytest
import torch.nn as nn
from omegaconf import OmegaConf

from pytorch_research_template.models import model_factory


def test_load_model_dispatches_to_registered_builder(monkeypatch) -> None:
    calls: list[object] = []

    def fake_builder(cfg: object) -> nn.Module:
        calls.append(cfg)
        return nn.Linear(1, 1)

    monkeypatch.setattr(model_factory, "_DISCOVERED", True)
    monkeypatch.setitem(model_factory.model_registry._entries, "cnn", fake_builder)

    cfg = OmegaConf.create({
        "model": {"name": "cnn"},
        "data": {"num_classes": 10},
    })
    model = model_factory.load_model(cfg)

    assert isinstance(model, nn.Linear)
    assert calls == [cfg]


def test_load_model_rejects_unknown_model() -> None:
    cfg = OmegaConf.create({"model": {"name": "unknown"}, "data": {"num_classes": 10}})

    with pytest.raises(ValueError, match="Unknown model"):
        model_factory.load_model(cfg)
