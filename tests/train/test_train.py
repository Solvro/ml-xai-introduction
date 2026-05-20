from __future__ import annotations

from dataclasses import dataclass, field

import torch
from omegaconf import OmegaConf

from ml_xai_introduction import train as train_module


@dataclass
class FakeLoader:
    batches: int
    yielded: bool = False

    def __len__(self) -> int:
        return self.batches

    def __iter__(self):
        if self.yielded:
            return iter(())

        self.yielded = True
        image = torch.zeros((2, 1, 28, 28), dtype=torch.float32)
        labels = torch.tensor([1, 2], dtype=torch.int64)
        return iter([(image, labels)])


@dataclass
class FakeTracker:
    params_calls: list[dict[str, object]] = field(default_factory=list)
    metrics_calls: list[dict[str, object]] = field(default_factory=list)
    finished: bool = False

    def log_params(self, params: dict[str, object]) -> None:
        self.params_calls.append(params)

    def log_metrics(self, metrics: dict[str, object], step: int | None = None) -> None:
        self.metrics_calls.append(metrics)

    def finish(self) -> None:
        self.finished = True


def test_main_logs_config_dataset_and_finish(monkeypatch, capsys) -> None:
    cfg = OmegaConf.create(
        {
            "paths": {"output_dir": "outputs/default", "checkpoint_dir": "outputs/default/checkpoints"},
            "logging": {"backends": ["none"]},
            "model": {"name": "baseline", "num_classes": 10, "input_shape": [1, 28, 28]},
            "data": {"name": "mnist", "root": "data/mnist"},
        }
    )
    fake_tracker = FakeTracker()

    monkeypatch.setattr(train_module, "_load_config", lambda: cfg)
    monkeypatch.setattr(train_module, "build_tracker", lambda logging_cfg: fake_tracker)
    monkeypatch.setattr(train_module, "dataset", lambda cfg: (FakeLoader(3), FakeLoader(2)))

    train_module.main()

    output = capsys.readouterr().out
    assert "Train batches: 3" in output
    assert "Test batches: 2" in output
    assert "First train batch images shape: (2, 1, 28, 28)" in output
    assert "First train batch labels shape: (2,)" in output
    assert fake_tracker.finished is True
    assert fake_tracker.params_calls[0]["data"]["name"] == "mnist"
    assert fake_tracker.metrics_calls[0] == {
        "data/train_batches": 3,
        "data/test_batches": 2,
        "data/image_height": 28,
        "data/image_width": 28,
    }