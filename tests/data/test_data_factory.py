from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from ml_xai_introduction.data import data_factory
from ml_xai_introduction.data.data_base import DataBundle


def _minimal_cfg(dataset_name: str, tmp_path: Path) -> object:
    return OmegaConf.create({
        "data": {
            "name": dataset_name,
            "root": str(tmp_path / dataset_name),
            "num_classes": 10,
            "val_size": 2,
            "download": False,
            "normalize": True,
            "num_workers": 0,
            "pin_memory": False,
        },
        "training": {
            "seed": 42,
            "batch_size": 2,
            "shuffle": False,
        },
    })


def test_load_data_dispatches_to_registered_builder(monkeypatch, tmp_path: Path) -> None:
    calls: list[object] = []

    def fake_builder(cfg: object) -> DataBundle:
        calls.append(cfg)
        return DataBundle(train="train", val="val", test="test")  # type: ignore[arg-type]

    monkeypatch.setattr(data_factory, "_DISCOVERED", True)
    monkeypatch.setitem(data_factory.dataset_registry._entries, "mnist", fake_builder)

    cfg = _minimal_cfg("mnist", tmp_path)
    bundle = data_factory.load_data(cfg)
    assert bundle.train == "train"
    assert bundle.val == "val"
    assert bundle.test == "test"
    assert calls == [cfg]


def test_load_data_normalizes_dataset_name(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(data_factory, "_DISCOVERED", True)
    monkeypatch.setitem(
        data_factory.dataset_registry._entries,
        "fashion_mnist",
        lambda cfg: DataBundle(train="t", val="v", test="te"),  # type: ignore[arg-type]
    )

    cfg = _minimal_cfg("fashion-mnist", tmp_path)
    bundle = data_factory.load_data(cfg)
    assert bundle.train == "t"
    assert bundle.val == "v"
    assert bundle.test == "te"


def test_load_data_rejects_unknown_dataset(tmp_path: Path) -> None:
    cfg = _minimal_cfg("unknown_dataset", tmp_path)

    with pytest.raises(ValueError, match="Unknown dataset"):
        data_factory.load_data(cfg)
