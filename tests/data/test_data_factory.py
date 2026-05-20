from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from ml_xai_introduction.data import data_factory


def test_dataset_factory_dispatches_to_mnist(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_mnist_loaders(**kwargs):
        calls.append(kwargs)
        return "train", "test"

    monkeypatch.setattr(data_factory, "get_mnist_loaders", fake_mnist_loaders)

    cfg = OmegaConf.create({
        "data": {
            "name": "mnist",
            "root": str(tmp_path / "mnist"),
            "batch_size": 16,
            "download": True,
            "normalize": False,
            "num_workers": 2,
            "pin_memory": True,
            "shuffle_train": False,
        }
    })

    assert data_factory.dataset(cfg) == ("train", "test")
    assert calls == [
        {
            "root": str(tmp_path / "mnist"),
            "batch_size": 16,
            "download": True,
            "normalize": False,
            "num_workers": 2,
            "pin_memory": True,
            "shuffle_train": False,
        }
    ]


def test_dataset_factory_dispatches_to_fashion_mnist(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_fashion_loaders(**kwargs):
        calls.append(kwargs)
        return "fashion-train", "fashion-test"

    monkeypatch.setattr(data_factory, "get_fashion_mnist_loaders", fake_fashion_loaders)

    cfg = OmegaConf.create({
        "data": {
            "name": "fashion-mnist",
            "root": str(tmp_path / "fashion_mnist"),
            "batch_size": 32,
            "download": False,
            "normalize": True,
            "num_workers": 0,
            "pin_memory": False,
            "shuffle_train": True,
        }
    })

    assert data_factory.dataset(cfg) == ("fashion-train", "fashion-test")
    assert calls == [
        {
            "root": str(tmp_path / "fashion_mnist"),
            "batch_size": 32,
            "download": False,
            "normalize": True,
            "num_workers": 0,
            "pin_memory": False,
            "shuffle_train": True,
        }
    ]


def test_dataset_factory_rejects_unknown_dataset() -> None:
    cfg = OmegaConf.create({"data": {"name": "cifar10"}})

    with pytest.raises(ValueError, match="Unsupported dataset"):
        data_factory.dataset(cfg)
