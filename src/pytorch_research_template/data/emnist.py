"""EMNIST dataset plugin."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from pytorch_research_template.data.data_base import DataBundle
from pytorch_research_template.data.data_factory import dataset_registry


def _build_transform() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1751,), (0.3267,))])


def _create_generator(seed: int | None) -> torch.Generator | None:
    return torch.Generator().manual_seed(seed) if seed is not None else None


def _split_train_val(
    train_dataset: datasets.EMNIST,
    val_size: int,
    generator: torch.Generator | None,
) -> tuple[Subset, Subset]:
    train_set_size = len(train_dataset) - val_size
    return random_split(train_dataset, [train_set_size, val_size], generator=generator)


@dataset_registry.register("emnist")
def build_emnist(cfg: DictConfig) -> DataBundle:
    root = Path(cfg.data.root)
    generator = _create_generator(cfg.training.seed if cfg.training.seed is not None else None)
    transform = _build_transform()

    train_dataset = datasets.EMNIST(
        root=root,
        split="balanced",
        train=True,
        download=bool(cfg.data.download),
        transform=transform,
    )
    test_dataset = datasets.EMNIST(
        root=root,
        split="balanced",
        train=False,
        download=bool(cfg.data.download),
        transform=transform,
    )
    train_subset, val_subset = _split_train_val(train_dataset, int(cfg.data.val_size), generator)

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
