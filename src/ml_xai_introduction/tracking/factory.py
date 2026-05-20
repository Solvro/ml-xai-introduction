"""Factory for config-driven tracking backends."""

from __future__ import annotations

import importlib
from collections.abc import Sequence

from omegaconf import DictConfig

from .base import TrackingManager

_BACKEND_MODULES = {
    "wandb": ".wandb",
    "w_and_b": ".wandb",
    "mlflow": ".mlflow",
    "tensorboard": ".tensorboard",
}

_SKIP_BACKENDS = {"", "none", "null", "disabled"}


def _normalize_backend_names(raw_backends: Sequence[object] | str | None) -> list[str]:
    if raw_backends is None:
        return []

    if isinstance(raw_backends, str):
        raw_backends = [raw_backends]

    normalized: list[str] = []
    for backend in raw_backends:
        backend_name = str(backend).strip().lower().replace("-", "_")
        if backend_name in _SKIP_BACKENDS:
            continue
        normalized.append(backend_name)

    return normalized


def build_tracker(cfg: DictConfig) -> TrackingManager:
    """Build a composite tracker from `cfg.backends`."""

    backends = []
    for backend_name in _normalize_backend_names(cfg.get("backends", [])):
        module_name = _BACKEND_MODULES.get(backend_name)
        if module_name is None:
            raise ValueError(f"Unsupported tracking backend: {backend_name}")

        module = importlib.import_module(module_name, package=__package__)
        backends.append(module.build_tracker(cfg))

    return TrackingManager(tuple(backends))
