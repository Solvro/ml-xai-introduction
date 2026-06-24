from __future__ import annotations

from dataclasses import dataclass, field

from omegaconf import OmegaConf

from ml_xai_introduction.metrics.metric_logger import MetricLogger


@dataclass
class FakeTracker:
    metrics_calls: list[tuple[dict[str, float], int | None]] = field(default_factory=list)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.metrics_calls.append((metrics, step))

    def log_params(self, params: dict[str, object]) -> None:
        pass

    def log_artifact(self, path: str, name: str | None = None) -> None:
        pass

    def finish(self) -> None:
        pass


def test_metric_logger_applies_phase_prefixes() -> None:
    tracker = FakeTracker()
    logger = MetricLogger(tracker, OmegaConf.create({}))

    logger.log_run_start({"data/train_batches": 3.0})
    logger.log_epoch("train", {"loss": 0.5}, step=1)
    logger.log_epoch("validation", {"accuracy": 0.9}, step=1)
    logger.log_eval({"f1": 0.8}, step=11)
    logger.log_summary({"best_val_loss": 0.4})

    assert tracker.metrics_calls[0] == ({"data/train_batches": 3.0}, 0)
    assert tracker.metrics_calls[1] == ({"train/loss": 0.5}, 1)
    assert tracker.metrics_calls[2] == ({"val/accuracy": 0.9}, 1)
    assert tracker.metrics_calls[3] == ({"test/f1": 0.8}, 11)
    assert tracker.metrics_calls[4] == ({"summary/best_val_loss": 0.4}, None)
