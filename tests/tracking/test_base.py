from __future__ import annotations

from ml_xai_introduction.tracking.base import TrackingManager


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


def test_tracking_manager_with_no_backends_is_noop() -> None:
    tracker = TrackingManager()

    tracker.log_metrics({"accuracy": 1.0}, step=1)
    tracker.log_params({"batch_size": 64})
    tracker.log_artifact("artifact.txt")
    tracker.finish()

    assert tracker.backends == ()


def test_tracking_manager_dispatches_to_all_backends() -> None:
    backend_one = DummyBackend()
    backend_two = DummyBackend()
    tracker = TrackingManager((backend_one, backend_two))

    tracker.log_metrics({"loss": 0.5}, step=2)
    tracker.log_params({"epochs": 10})
    tracker.log_artifact("model.bin", name="model")
    tracker.finish()

    assert backend_one.metrics_calls == [({"loss": 0.5}, 2)]
    assert backend_two.metrics_calls == [({"loss": 0.5}, 2)]
    assert backend_one.params_calls == [{"epochs": 10}]
    assert backend_two.params_calls == [{"epochs": 10}]
    assert backend_one.artifact_calls == [("model.bin", "model")]
    assert backend_two.artifact_calls == [("model.bin", "model")]
    assert backend_one.finished is True
    assert backend_two.finished is True