"""Weights & Biases adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from omegaconf import DictConfig, OmegaConf

from ml_xai_introduction.tracking.tracking_factory import tracking_registry


def _require_wandb() -> ModuleType:
    try:
        import wandb
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("tracking backend 'wandb' requires the 'wandb' package") from exc

    return wandb


@dataclass(slots=True)
class WandbTracker:
    cfg: DictConfig
    _run: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        wandb = _require_wandb()
        self._run = wandb.init(
            project=str(self.cfg.project_name),
            name=str(self.cfg.experiment_name),
            mode=str(self.cfg.wandb.mode),
            entity=self.cfg.wandb.entity,
            tags=list(self.cfg.wandb.tags),
            notes=self.cfg.wandb.notes,
            config=OmegaConf.to_container(self.cfg, resolve=True),
        )

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self._run.log(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        self._run.config.update(params, allow_val_change=True)

    def log_artifact(self, path: str, name: str | None = None) -> None:
        wandb = _require_wandb()
        artifact = wandb.Artifact(name or Path(path).name, type="artifact")
        artifact.add_file(path)
        self._run.log_artifact(artifact)

    def finish(self) -> None:
        self._run.finish()


@tracking_registry.register("wandb", "w_and_b")
def build_tracker(cfg: DictConfig) -> WandbTracker:
    return WandbTracker(cfg=cfg)
