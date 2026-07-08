"""Generic plugin registry with lazy auto-discovery."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")

SKIP_REGISTRY_NAMES = frozenset({"", "none", "null", "disabled"})


def normalize_name(raw: object) -> str:
    """Normalize config/plugin names: strip, lowercase, hyphen to underscore."""

    return str(raw).strip().lower().replace("-", "_")


def is_skipped_name(name: str) -> bool:
    return normalize_name(name) in SKIP_REGISTRY_NAMES


class Registry(Generic[T]):
    def __init__(self, label: str) -> None:
        self._label = label
        self._entries: dict[str, Callable[..., T]] = {}

    def register(self, *names: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(build_function: Callable[..., T]) -> Callable[..., T]:
            for name in names:
                self._entries[normalize_name(name)] = build_function
            return build_function

        return decorator

    def get(self, name: str) -> Callable[..., T]:
        normalized = normalize_name(name)
        if normalized not in self._entries:
            raise ValueError(f"Unknown {self._label}: {name}")
        return self._entries[normalized]


def autodiscover(package_name: str, exclude: frozenset[str]) -> None:
    package = importlib.import_module(package_name)
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return

    prefix = f"{package_name}."
    for module_info in pkgutil.iter_modules(package_path):
        if module_info.name.startswith("_") or module_info.name in exclude:
            continue
        importlib.import_module(prefix + module_info.name)
