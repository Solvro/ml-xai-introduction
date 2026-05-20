"""MNIST dataset helpers backed by torchvision.

The loader checks whether the raw MNIST files already exist in the target
directory and downloads them only when they are missing.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MNIST_DEFAULT_ROOT = Path("data") / "mnist"
MNIST_RAW_FILES = (
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
)


def _build_transform(normalize: bool = True) -> transforms.Compose:
    steps: list[object] = [transforms.ToTensor()]
    if normalize:
        steps.append(transforms.Normalize((0.1307,), (0.3081,)))
    return transforms.Compose(steps)


def _mnist_raw_dir(root: Path) -> Path:
    return root / "MNIST" / "raw"


def _mnist_is_present(root: Path) -> bool:
    raw_dir = _mnist_raw_dir(root)
    return all((raw_dir / file_name).exists() for file_name in MNIST_RAW_FILES)


def _ensure_mnist_available(root: Path) -> None:
    if _mnist_is_present(root):
        print(f"MNIST dataset found at {root}; using existing files.")
        return

    print(f"MNIST dataset not found at {root}; downloading to that location.")
    # Instantiating the dataset with download=True triggers torchvision's
    # built-in MNIST download path.
    datasets.MNIST(root=root, train=True, download=True)
    print(f"MNIST dataset downloaded to {root}.")


def get_mnist_datasets(
    root: str | Path = MNIST_DEFAULT_ROOT,
    download: bool = True,
    normalize: bool = True,
) -> tuple[datasets.MNIST, datasets.MNIST]:
    """Return train and test MNIST datasets."""

    root_path = Path(root)
    if download:
        _ensure_mnist_available(root_path)

    transform = _build_transform(normalize=normalize)
    train_dataset = datasets.MNIST(root=root_path, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root=root_path, train=False, download=False, transform=transform)
    return train_dataset, test_dataset


def get_mnist_loaders(
    root: str | Path = MNIST_DEFAULT_ROOT,
    batch_size: int = 64,
    download: bool = True,
    normalize: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle_train: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Return train and test dataloaders for MNIST."""

    train_dataset, test_dataset = get_mnist_datasets(root=root, download=download, normalize=normalize)
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
