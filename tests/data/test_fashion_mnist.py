from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from pytorch_research_template.data import fashion_mnist


class TinyDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        image = torch.full((1, 2, 2), float(index + 1), dtype=torch.float32)
        return image, index % 10


class FakeFashionMNIST(TinyDataset):
    init_calls: list[dict[str, object]] = []

    def __init__(self, root: str | Path, train: bool, download: bool, transform=None) -> None:
        FakeFashionMNIST.init_calls.append({
            "root": Path(root),
            "train": train,
            "download": download,
            "transform": transform,
        })
        self.transform = transform


def test_fashion_mnist_presence_detection(tmp_fashion_mnist_root: Path) -> None:
    raw_dir = tmp_fashion_mnist_root / "FashionMNIST" / "raw"
    raw_dir.mkdir(parents=True)
    for file_name in fashion_mnist.FASHION_MNIST_RAW_FILES:
        (raw_dir / file_name).write_text("ok")

    assert fashion_mnist._is_present(tmp_fashion_mnist_root) is True


def test_fashion_mnist_ensure_downloads_when_missing(monkeypatch, tmp_fashion_mnist_root: Path) -> None:
    calls: list[tuple[Path, bool, bool]] = []

    def fake_fashion_mnist(root: str | Path, train: bool, download: bool, transform=None):
        calls.append((Path(root), train, download))
        return object()

    monkeypatch.setattr(fashion_mnist.datasets, "FashionMNIST", fake_fashion_mnist)

    fashion_mnist._ensure_available(tmp_fashion_mnist_root)

    assert calls == [(tmp_fashion_mnist_root, True, True)]


def test_get_fashion_mnist_loaders_returns_batched_tensors(monkeypatch, tmp_fashion_mnist_root: Path) -> None:
    monkeypatch.setattr(fashion_mnist, "get_fashion_mnist_datasets", lambda **kwargs: (TinyDataset(), TinyDataset()))

    train_loader, test_loader = fashion_mnist.get_fashion_mnist_loaders(
        root=tmp_fashion_mnist_root, batch_size=2, download=False
    )
    images, labels = next(iter(train_loader))

    assert len(train_loader) == 2
    assert len(test_loader) == 2
    assert images.shape == torch.Size([2, 1, 2, 2])
    assert labels.shape == torch.Size([2])
