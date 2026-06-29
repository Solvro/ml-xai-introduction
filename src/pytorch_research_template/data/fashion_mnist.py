"""FashionMNIST dataset helpers backed by torchvision."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import datasets

from pytorch_research_template.data.data_base import (
    DataBundle,
    build_loaders,
    build_transform,
    create_generator,
    ensure_dataset_available,
    split_train_val,
)
from pytorch_research_template.data.data_factory import dataset_registry

FASHION_MNIST_DEFAULT_ROOT = Path("data") / "fashion_mnist"
FASHION_MNIST_RAW_FILES = (
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
)


def _ensure_fashion_mnist_available(root: Path) -> None:
    ensure_dataset_available(root, "FashionMNIST", datasets.FashionMNIST, FASHION_MNIST_RAW_FILES, "FashionMNIST")


def get_fashion_mnist_datasets(
    root: str | Path = FASHION_MNIST_DEFAULT_ROOT,
    download: bool = True,
    normalize: bool = True,
) -> tuple[datasets.FashionMNIST, datasets.FashionMNIST]:
    root_path = Path(root)
    if download:
        _ensure_fashion_mnist_available(root_path)

    transform = build_transform(0.2860, 0.3530, normalize=normalize)
    train_dataset = datasets.FashionMNIST(root=root_path, train=True, download=False, transform=transform)
    test_dataset = datasets.FashionMNIST(root=root_path, train=False, download=False, transform=transform)
    return train_dataset, test_dataset


def get_fashion_mnist_loaders(
    root: str | Path = FASHION_MNIST_DEFAULT_ROOT,
    batch_size: int = 64,
    download: bool = True,
    normalize: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle_train: bool = True,
) -> tuple[DataLoader, DataLoader]:
    train_dataset, test_dataset = get_fashion_mnist_datasets(root=root, download=download, normalize=normalize)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


@dataset_registry.register("fashion_mnist", "fashion-mnist")
def build_fashion_mnist(cfg: DictConfig) -> DataBundle:
    root = Path(cfg.data.root)
    if bool(cfg.data.download):
        _ensure_fashion_mnist_available(root)

    generator = create_generator(cfg.training.seed if cfg.training.seed is not None else None)
    transform = build_transform(0.2860, 0.3530, normalize=bool(cfg.data.normalize))
    train_dataset = datasets.FashionMNIST(root=root, train=True, download=False, transform=transform)
    test_dataset = datasets.FashionMNIST(root=root, train=False, download=False, transform=transform)
    train_subset, val_subset = split_train_val(train_dataset, int(cfg.data.val_size), generator)

    return build_loaders(cfg, train_subset, val_subset, test_dataset, generator)
