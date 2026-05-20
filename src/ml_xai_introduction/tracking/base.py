"""Shared tracking interfaces and composite dispatcher."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


class TrackingBackend(Protocol):
    """Common contract for optional tracking backends."""

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log scalar or structured metrics."""

    def log_params(self, params: dict[str, Any]) -> None:
        """Log run parameters or configuration."""

    def log_artifact(self, path: str, name: str | None = None) -> None:
        """Log a file artifact when supported."""

    def finish(self) -> None:
        """Flush and close the backend."""


@dataclass(slots=True)
class TrackingManager:
    """Dispatches tracking calls to all configured backends."""

    backends: Sequence[TrackingBackend] = field(default_factory=tuple)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        for backend in self.backends:
            backend.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        for backend in self.backends:
            backend.log_params(params)

    def log_artifact(self, path: str, name: str | None = None) -> None:
        for backend in self.backends:
            backend.log_artifact(path, name=name)

    def finish(self) -> None:
        for backend in self.backends:
            backend.finish()
