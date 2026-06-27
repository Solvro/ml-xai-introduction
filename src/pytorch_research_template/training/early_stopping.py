"""Early stopping helper."""

from __future__ import annotations


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float, monitor: str = "val_loss") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self._higher_is_better = "accuracy" in monitor
        self.best: float | None = None
        self.counter = 0

    def step(self, metrics: dict[str, float]) -> bool:
        value = metrics[self.monitor]

        if self.best is None:
            self.best = value
            return False

        improved = value > self.best + self.min_delta if self._higher_is_better else value < self.best - self.min_delta

        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience
