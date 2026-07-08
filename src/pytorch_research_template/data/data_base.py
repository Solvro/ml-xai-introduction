"""Dataset plugin contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms


@dataclass(frozen=True, slots=True)
class DataBundle:
    """Train, validation, and test dataloaders returned by a dataset plugin."""

    train: DataLoader
    val: DataLoader
    test: DataLoader


LoadDataFn = Callable[[DictConfig], DataBundle]


def create_generator(seed: int | None) -> torch.Generator | None:
    return torch.Generator().manual_seed(seed) if seed is not None else None


def build_transform(mean: float, std: float, channels: int = 1, normalize: bool = True) -> transforms.Compose:
    steps: list[object] = [transforms.ToTensor()]
    if normalize:
        means = tuple(mean for _ in range(channels))
        stds = tuple(std for _ in range(channels))
        steps.append(transforms.Normalize(means, stds))
    return transforms.Compose(steps)


def split_train_val(
    train_dataset: Dataset,
    val_size: int,
    generator: torch.Generator | None,
) -> list[Subset]:
    train_set_size = len(train_dataset) - val_size
    return random_split(train_dataset, [train_set_size, val_size], generator=generator)


def ensure_dataset_available(
    root: Path,
    subfolder: str,
    dataset_cls: type,
    raw_files: tuple[str, ...],
    name: str,
) -> None:
    raw_dir = root / subfolder / "raw"
    if all((raw_dir / file_name).exists() for file_name in raw_files):
        print(f"{name} dataset found at {root}; using existing files.")
        return
    print(f"{name} dataset not found at {root}; downloading to that location.")
    dataset_cls(root=root, train=True, download=True)
    print(f"{name} dataset downloaded to {root}.")


def build_loaders(
    cfg: DictConfig, train_subset: Subset, val_subset: Subset, test_dataset: Dataset, generator: torch.Generator | None
) -> DataBundle:
    loader_kwargs = {
        "batch_size": int(cfg.training.batch_size),
        "num_workers": int(cfg.data.num_workers),
        "pin_memory": bool(cfg.data.pin_memory),
    }

    train_loader = DataLoader(
        train_subset,
        shuffle=bool(cfg.training.shuffle),
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return DataBundle(train=train_loader, val=val_loader, test=test_loader)
