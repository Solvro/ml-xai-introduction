"""Metric logging to trackers and console."""

from __future__ import annotations

from omegaconf import DictConfig

from pytorch_research_template.tracking.tracking_base import TrackingBackend

PHASE_PREFIX = {
    "train": "train",
    "validation": "val",
    "test": "test",
}


class MetricLogger:
    def __init__(self, tracker: TrackingBackend, cfg: DictConfig) -> None:
        self._tracker = tracker
        self._cfg = cfg

    def log_run_start(self, metrics: dict[str, float]) -> None:
        self._tracker.log_metrics(metrics, step=0)

    def log_epoch(self, phase: str, metrics: dict[str, float], step: int) -> None:
        prefix = PHASE_PREFIX.get(phase, phase)
        keyed = {f"{prefix}/{name}": value for name, value in metrics.items()}
        self._tracker.log_metrics(keyed, step=step)

    def log_extra(self, metrics: dict[str, float], step: int) -> None:
        self._tracker.log_metrics(metrics, step=step)

    def log_eval(self, metrics: dict[str, float], step: int) -> None:
        keyed = {f"test/{name}": value for name, value in metrics.items()}
        self._tracker.log_metrics(keyed, step=step)

    def log_summary(self, metrics: dict[str, float]) -> None:
        keyed = {f"summary/{name}": value for name, value in metrics.items()}
        self._tracker.log_metrics(keyed)

    def print_epoch(
        self,
        epoch: int,
        epochs: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        learning_rate: float,
    ) -> None:
        train_loss = train_metrics.get("loss", float("nan"))
        train_acc = train_metrics.get("accuracy", float("nan"))
        val_loss = val_metrics.get("loss", float("nan"))
        val_acc = val_metrics.get("accuracy", float("nan"))
        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
            f"val loss: {val_loss:.4f} acc: {val_acc:.4f} | "
            f"lr: {learning_rate:.2e}"
        )

    def print_eval(self, metrics: dict[str, float]) -> None:
        loss = metrics.get("loss", float("nan"))
        acc = metrics.get("accuracy", float("nan"))
        print(f"Test | loss: {loss:.4f} acc: {acc:.4f}")
