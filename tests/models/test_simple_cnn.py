from __future__ import annotations

from omegaconf import OmegaConf

from pytorch_research_template.models.model_factory import load_model


def test_model_registry_builds_simple_cnn() -> None:
    cfg = OmegaConf.create({"model": {"name": "simple_cnn"}, "data": {"num_classes": 10}})
    model = load_model(cfg)
    assert model.__class__.__name__ == "SimpleCNN"


def test_model_registry_builds_residual_cnn() -> None:
    cfg = OmegaConf.create({"model": {"name": "residual_cnn"}, "data": {"num_classes": 10}})
    model = load_model(cfg)
    assert model.__class__.__name__ == "ResidualCNN"
