"""TensorBoard adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig

from ml_xai_introduction.tracking.tracking_factory import tracking_registry


def _require_tensorboard_writer() -> type[Any]:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("tracking backend 'tensorboard' requires TensorBoard support") from exc

    return SummaryWriter


@dataclass(slots=True)
class TensorBoardTracker:
    cfg: DictConfig
    _writer: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        summary_writer = _require_tensorboard_writer()
        self._writer = summary_writer(log_dir=str(self.cfg.tensorboard.log_dir))

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        for key, value in metrics.items():
            if isinstance(value, int | float):
                self._writer.add_scalar(
                    key,
                    value,
                    global_step=0 if step is None else step,
                )
            else:
                self._writer.add_text(
                    key,
                    json.dumps(value, sort_keys=True, default=str),
                    global_step=0 if step is None else step,
                )

    def log_params(self, params: dict[str, Any]) -> None:
        self._writer.add_text("params/config", json.dumps(params, indent=2, sort_keys=True, default=str))

    def log_artifact(self, path: str, name: str | None = None) -> None:
        artifact_name = name or path.split("/")[-1]
        self._writer.add_text(f"artifacts/{artifact_name}", path)

    def finish(self) -> None:
        self._writer.flush()
        self._writer.close()


@tracking_registry.register("tensorboard")
def build_tracker(cfg: DictConfig) -> TensorBoardTracker:
    return TensorBoardTracker(cfg=cfg)
