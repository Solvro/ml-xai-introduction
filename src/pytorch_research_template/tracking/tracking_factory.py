"""Factory for config-driven tracking backends."""

from __future__ import annotations

from collections.abc import Sequence

from omegaconf import DictConfig

from pytorch_research_template.registry import Registry, autodiscover, is_skipped_name, normalize_name
from pytorch_research_template.tracking.tracking_base import TrackingBackend, TrackingManager

tracking_registry: Registry[TrackingBackend] = Registry("tracking backend")

_DISCOVERED = False


def _ensure_discovered() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    autodiscover(
        "pytorch_research_template.tracking",
        frozenset({"tracking_factory", "tracking_base"}),
    )
    _DISCOVERED = True


def _normalize_backend_names(raw_backends: Sequence[object] | str | None) -> list[str]:
    if raw_backends is None:
        return []

    if isinstance(raw_backends, str):
        raw_backends = [raw_backends]

    normalized: list[str] = []
    for backend in raw_backends:
        backend_name = normalize_name(backend)
        if is_skipped_name(backend_name):
            continue
        normalized.append(backend_name)

    return normalized


def build_tracker(cfg: DictConfig) -> TrackingManager:
    """Build a composite tracker from `cfg.backends`."""

    _ensure_discovered()
    backends: list[TrackingBackend] = []
    for backend_name in _normalize_backend_names(cfg.get("backends", [])):
        build_fn = tracking_registry.get(backend_name)
        backends.append(build_fn(cfg))

    return TrackingManager(tuple(backends))
