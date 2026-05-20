"""MLflow adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig


def _require_mlflow():
    try:
        import mlflow
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("tracking backend 'mlflow' requires the 'mlflow' package") from exc

    return mlflow


def _flatten_params(params: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in params.items():
        flat_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_params(value, prefix=flat_key))
        elif isinstance(value, (list, tuple, set)):
            flattened[flat_key] = json.dumps(list(value), default=str)
        else:
            flattened[flat_key] = value
    return flattened


@dataclass(slots=True)
class MLflowTracker:
    cfg: DictConfig
    _run: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mlflow = _require_mlflow()
        tracking_uri = self.cfg.mlflow.tracking_uri
        if tracking_uri is not None:
            mlflow.set_tracking_uri(str(tracking_uri))

        mlflow.set_experiment(str(self.cfg.project_name))
        self._run = mlflow.start_run(run_name=str(self.cfg.experiment_name), nested=bool(self.cfg.mlflow.nested))

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        mlflow = _require_mlflow()
        mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow = _require_mlflow()
        mlflow.log_params(_flatten_params(params))

    def log_artifact(self, path: str, name: str | None = None) -> None:
        mlflow = _require_mlflow()
        mlflow.log_artifact(path, artifact_path=name)

    def finish(self) -> None:
        mlflow = _require_mlflow()
        mlflow.end_run()


def build_tracker(cfg: DictConfig) -> MLflowTracker:
    return MLflowTracker(cfg=cfg)