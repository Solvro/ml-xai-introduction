from __future__ import annotations

from omegaconf import OmegaConf

from pytorch_research_template.data.data_factory import load_data
from pytorch_research_template.models.model_factory import load_model
from pytorch_research_template.registry import Registry, normalize_name


def test_dataset_registry_builds_mnist_with_minimal_config(monkeypatch, tmp_path) -> None:
    import pytorch_research_template.data.mnist as mnist_module

    class FakeSubset:
        def __init__(self, size: int) -> None:
            self._size = size

        def __len__(self) -> int:
            return self._size

    monkeypatch.setattr(mnist_module, "_ensure_mnist_available", lambda root: None)
    monkeypatch.setattr(
        mnist_module,
        "_split_train_val",
        lambda train_dataset, val_size, generator: (FakeSubset(8), FakeSubset(val_size)),
    )
    monkeypatch.setattr(
        mnist_module.datasets,
        "MNIST",
        lambda **kwargs: FakeSubset(10),
    )

    cfg = OmegaConf.create({
        "data": {
            "name": "mnist",
            "root": str(tmp_path),
            "val_size": 2,
            "download": False,
            "normalize": True,
            "num_workers": 0,
            "pin_memory": False,
        },
        "training": {"seed": 1, "batch_size": 2, "shuffle": False},
    })

    data = load_data(cfg)
    assert len(data.train) == 4
    assert len(data.val) == 1
    assert len(data.test) == 5


def test_model_registry_builds_cnn() -> None:
    cfg = OmegaConf.create({"model": {"name": "cnn"}, "data": {"num_classes": 10}})
    model = load_model(cfg)
    assert model.__class__.__name__ == "CNN"


def test_model_registry_builds_baseline() -> None:
    cfg = OmegaConf.create({"model": {"name": "baseline"}, "data": {"num_classes": 10}})
    model = load_model(cfg)
    assert model.__class__.__name__ == "BaselineNN"


def test_normalize_name() -> None:
    assert normalize_name("Fashion-MNIST") == "fashion_mnist"
    assert normalize_name("  CNN  ") == "cnn"


def test_registry_supports_multiple_aliases() -> None:
    registry: Registry[str] = Registry("example")
    calls: list[str] = []

    @registry.register("foo", "bar-baz")
    def builder() -> str:
        calls.append("ok")
        return "built"

    assert registry.get("foo")() == "built"
    assert registry.get("bar-baz")() == "built"
