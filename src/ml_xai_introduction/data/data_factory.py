"""Dataset factory for config-driven data loading."""

from __future__ import annotations

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .fashion_mnist import get_fashion_mnist_loaders
from .mnist import get_mnist_loaders


def dataset(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    """Return train and test dataloaders for the dataset selected in cfg."""

    dataset_name = str(cfg.data.name).lower()

    if dataset_name == "mnist":
        return get_mnist_loaders(
            root=cfg.data.root,
            batch_size=int(cfg.data.batch_size),
            download=bool(cfg.data.download),
            normalize=bool(cfg.data.normalize),
            num_workers=int(cfg.data.num_workers),
            pin_memory=bool(cfg.data.pin_memory),
            shuffle_train=bool(cfg.data.shuffle_train),
        )

    if dataset_name in {"fashion_mnist", "fashion-mnist"}:
        return get_fashion_mnist_loaders(
            root=cfg.data.root,
            batch_size=int(cfg.data.batch_size),
            download=bool(cfg.data.download),
            normalize=bool(cfg.data.normalize),
            num_workers=int(cfg.data.num_workers),
            pin_memory=bool(cfg.data.pin_memory),
            shuffle_train=bool(cfg.data.shuffle_train),
        )

    raise ValueError(f"Unsupported dataset: {cfg.data.name}")