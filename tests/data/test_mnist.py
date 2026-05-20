from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from ml_xai_introduction.data import mnist


class TinyDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        image = torch.full((1, 2, 2), float(index + 1), dtype=torch.float32)
        return image, index % 10


class FakeMNIST(TinyDataset):
    init_calls: list[dict[str, object]] = []

    def __init__(self, root: str | Path, train: bool, download: bool, transform=None) -> None:
        FakeMNIST.init_calls.append({"root": Path(root), "train": train, "download": download, "transform": transform})
        self.transform = transform


def test_mnist_presence_detection(tmp_mnist_root: Path) -> None:
    raw_dir = tmp_mnist_root / "MNIST" / "raw"
    raw_dir.mkdir(parents=True)
    for file_name in mnist.MNIST_RAW_FILES:
        (raw_dir / file_name).write_text("ok")

    assert mnist._mnist_is_present(tmp_mnist_root) is True


def test_mnist_presence_detection_false_when_missing(tmp_mnist_root: Path) -> None:
    assert mnist._mnist_is_present(tmp_mnist_root) is False


def test_mnist_ensure_downloads_when_missing(monkeypatch, tmp_mnist_root: Path) -> None:
    calls: list[tuple[Path, bool, bool]] = []

    def fake_mnist(root: str | Path, train: bool, download: bool, transform=None):
        calls.append((Path(root), train, download))
        return SimpleNamespace()

    monkeypatch.setattr(mnist.datasets, "MNIST", fake_mnist)

    mnist._ensure_mnist_available(tmp_mnist_root)

    assert calls == [(tmp_mnist_root, True, True)]


def test_get_mnist_datasets_uses_download_check(monkeypatch, tmp_mnist_root: Path) -> None:
    FakeMNIST.init_calls.clear()
    monkeypatch.setattr(mnist.datasets, "MNIST", FakeMNIST)
    monkeypatch.setattr(mnist, "_ensure_mnist_available", lambda root: None)

    train_dataset, test_dataset = mnist.get_mnist_datasets(root=tmp_mnist_root, download=True, normalize=False)

    assert isinstance(train_dataset, TinyDataset)
    assert isinstance(test_dataset, TinyDataset)
    assert FakeMNIST.init_calls == [
        {"root": tmp_mnist_root, "train": True, "download": False, "transform": train_dataset.transform},
        {"root": tmp_mnist_root, "train": False, "download": False, "transform": test_dataset.transform},
    ]


def test_get_mnist_loaders_returns_batched_tensors(monkeypatch, tmp_mnist_root: Path) -> None:
    monkeypatch.setattr(mnist, "get_mnist_datasets", lambda **kwargs: (TinyDataset(), TinyDataset()))

    train_loader, test_loader = mnist.get_mnist_loaders(root=tmp_mnist_root, batch_size=2, download=False)
    images, labels = next(iter(train_loader))

    assert len(train_loader) == 2
    assert len(test_loader) == 2
    assert images.shape == torch.Size([2, 1, 2, 2])
    assert labels.shape == torch.Size([2])
