from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from pytorch_research_template.tracking import tracking_factory as factory
from pytorch_research_template.tracking.tracking_base import TrackingManager


class DummyBackend:
    def __init__(self) -> None:
        self.metrics_calls: list[tuple[dict[str, object], int | None]] = []
        self.params_calls: list[dict[str, object]] = []
        self.artifact_calls: list[tuple[str, str | None]] = []
        self.finished = False

    def log_metrics(self, metrics: dict[str, object], step: int | None = None) -> None:
        self.metrics_calls.append((metrics, step))

    def log_params(self, params: dict[str, object]) -> None:
        self.params_calls.append(params)

    def log_artifact(self, path: str, name: str | None = None) -> None:
        self.artifact_calls.append((path, name))

    def finish(self) -> None:
        self.finished = True


def test_build_tracker_returns_empty_manager_for_none_backend() -> None:
    cfg = OmegaConf.create({"backends": ["none"]})

    tracker = factory.build_tracker(cfg)

    assert isinstance(tracker, TrackingManager)
    assert tracker.backends == ()


def test_build_tracker_dispatches_via_registry(monkeypatch) -> None:
    backend_one = DummyBackend()
    backend_two = DummyBackend()

    monkeypatch.setattr(factory, "_DISCOVERED", True)

    def wandb_builder(cfg: object) -> DummyBackend:
        return backend_one

    monkeypatch.setitem(factory.tracking_registry._entries, "wandb", wandb_builder)
    monkeypatch.setitem(factory.tracking_registry._entries, "w_and_b", wandb_builder)
    monkeypatch.setitem(
        factory.tracking_registry._entries,
        "tensorboard",
        lambda cfg: backend_two,
    )

    cfg = OmegaConf.create({"backends": ["w_and_b", "tensorboard"]})
    tracker = factory.build_tracker(cfg)

    tracker.log_metrics({"loss": 0.5}, step=2)
    tracker.log_params({"epochs": 10})
    tracker.log_artifact("model.bin", name="model")
    tracker.finish()

    assert tracker.backends == (backend_one, backend_two)
    assert backend_one.metrics_calls == [({"loss": 0.5}, 2)]
    assert backend_two.metrics_calls == [({"loss": 0.5}, 2)]
    assert backend_one.params_calls == [{"epochs": 10}]
    assert backend_two.params_calls == [{"epochs": 10}]
    assert backend_one.artifact_calls == [("model.bin", "model")]
    assert backend_two.artifact_calls == [("model.bin", "model")]
    assert backend_one.finished is True
    assert backend_two.finished is True


def test_build_tracker_rejects_unknown_backend() -> None:
    cfg = OmegaConf.create({"backends": ["something-else"]})

    with pytest.raises(ValueError, match="Unknown tracking backend"):
        factory.build_tracker(cfg)
