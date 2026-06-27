"""Dataset plugin contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from omegaconf import DictConfig
from torch.utils.data import DataLoader


@dataclass(frozen=True, slots=True)
class DataBundle:
    """Train, validation, and test dataloaders returned by a dataset plugin."""

    train: DataLoader
    val: DataLoader
    test: DataLoader


LoadDataFn = Callable[[DictConfig], DataBundle]
