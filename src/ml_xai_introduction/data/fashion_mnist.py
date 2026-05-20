"""FashionMNIST dataset helpers backed by torchvision."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

FASHION_MNIST_DEFAULT_ROOT = Path("data") / "fashion_mnist"
FASHION_MNIST_RAW_FILES = (
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
)


def _build_transform(normalize: bool = True) -> transforms.Compose:
    steps: list[object] = [transforms.ToTensor()]
    if normalize:
        steps.append(transforms.Normalize((0.2860,), (0.3530,)))
    return transforms.Compose(steps)


def _raw_dir(root: Path) -> Path:
    return root / "FashionMNIST" / "raw"


def _is_present(root: Path) -> bool:
    raw_dir = _raw_dir(root)
    return all((raw_dir / file_name).exists() for file_name in FASHION_MNIST_RAW_FILES)


def _ensure_available(root: Path) -> None:
    if _is_present(root):
        print(f"FashionMNIST dataset found at {root}; using existing files.")
        return

    print(f"FashionMNIST dataset not found at {root}; downloading to that location.")
    datasets.FashionMNIST(root=root, train=True, download=True)
    print(f"FashionMNIST dataset downloaded to {root}.")


def get_fashion_mnist_datasets(
    root: str | Path = FASHION_MNIST_DEFAULT_ROOT,
    download: bool = True,
    normalize: bool = True,
) -> tuple[datasets.FashionMNIST, datasets.FashionMNIST]:
    root_path = Path(root)
    if download:
        _ensure_available(root_path)

    transform = _build_transform(normalize=normalize)
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
