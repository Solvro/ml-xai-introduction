from __future__ import annotations

from types import SimpleNamespace

from omegaconf import OmegaConf

from ml_xai_introduction.tracking.base import TrackingManager
from ml_xai_introduction.tracking import factory


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
    tracker.log_metrics({"accuracy": 1.0}, step=1)
    tracker.log_params({"batch_size": 64})
    tracker.log_artifact("artifact.txt")
    tracker.finish()
    assert tracker.backends == ()



def test_build_tracker_dispatches_to_all_backends(monkeypatch) -> None:
    backend_one = DummyBackend()
    backend_two = DummyBackend()

    def fake_import(module_name: str, package: str | None = None):
        if module_name == ".wandb":
            return SimpleNamespace(build_tracker=lambda cfg: backend_one)
        if module_name == ".tensorboard":
            return SimpleNamespace(build_tracker=lambda cfg: backend_two)
        raise AssertionError(f"Unexpected import: {module_name}")

    monkeypatch.setattr(factory.importlib, "import_module", fake_import)

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
