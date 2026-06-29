"""EMNIST dataset plugin."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig
from torchvision import datasets

from pytorch_research_template.data.data_base import (
    DataBundle,
    build_loaders,
    build_transform,
    create_generator,
    split_train_val,
)
from pytorch_research_template.data.data_factory import dataset_registry


@dataset_registry.register("emnist")
def build_emnist(cfg: DictConfig) -> DataBundle:
    root = Path(cfg.data.root)
    generator = create_generator(cfg.training.seed if cfg.training.seed is not None else None)
    transform = build_transform(0.1751, 0.3267, normalize=bool(cfg.data.normalize))

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
    train_subset, val_subset = split_train_val(train_dataset, int(cfg.data.val_size), generator)

    return build_loaders(cfg, train_subset, val_subset, test_dataset, generator)
