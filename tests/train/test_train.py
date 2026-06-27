from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from pytorch_research_template import train as train_module
from pytorch_research_template.data.data_base import DataBundle
from pytorch_research_template.metrics.metric_base import MetricContext


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
    metrics_calls: list[tuple[dict[str, object], int | None]] = field(default_factory=list)
    finished: bool = False

    def log_params(self, params: dict[str, object]) -> None:
        self.params_calls.append(params)

    def log_metrics(self, metrics: dict[str, object], step: int | None = None) -> None:
        self.metrics_calls.append((metrics, step))

    def finish(self) -> None:
        self.finished = True


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.view(x.size(0), -1))


def _fake_context() -> MetricContext:
    return MetricContext(
        predictions=torch.tensor([0, 1]),
        targets=torch.tensor([0, 1]),
        loss_sum=1.0,
        num_samples=2,
    )


def _cfg(tmp_path: Path) -> object:
    return OmegaConf.create({
        "paths": {
            "output_dir": str(tmp_path / "outputs"),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        },
        "logging": {"backends": ["none"]},
        "model": {"name": "cnn"},
        "data": {"name": "mnist", "num_classes": 10},
        "loss": {"name": "cross_entropy"},
        "metrics": {
            "train": {"active": ["loss"]},
            "validation": {"active": ["loss", "accuracy"]},
            "test": {"active": ["loss", "accuracy"]},
        },
        "training": {
            "epochs": 1,
            "seed": 42,
            "batch_size": 2,
            "shuffle": False,
            "optimizer": {"name": "adam", "lr": 1e-3, "weight_decay": 0.0, "momentum": 0.9},
            "scheduler": {"name": "none"},
            "early_stopping": {"enabled": False, "patience": 3, "min_delta": 1e-3, "monitor": "loss"},
            "export": {"onnx": False},
        },
    })


def test_main_runs_training_and_logs_metrics(monkeypatch, tmp_path: Path, capsys) -> None:
    cfg = _cfg(tmp_path)
    fake_tracker = FakeTracker()
    fake_ctx = _fake_context()
    checkpoint_path = Path(cfg.paths.checkpoint_dir) / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.touch()

    monkeypatch.setattr(train_module, "_load_config", lambda: cfg)
    monkeypatch.setattr(train_module, "build_tracker", lambda logging_cfg: fake_tracker)
    monkeypatch.setattr(
        train_module,
        "load_data",
        lambda cfg: DataBundle(train=FakeLoader(3), val=FakeLoader(2), test=FakeLoader(1)),
    )
    monkeypatch.setattr(train_module, "load_model", lambda cfg: TinyModel())
    monkeypatch.setattr(
        train_module,
        "setup_training",
        lambda model, cfg, steps_per_epoch=None: (
            torch.optim.Adam(model.parameters(), lr=1e-3),
            nn.CrossEntropyLoss(),
            None,
        ),
    )
    monkeypatch.setattr(train_module, "run_train_epoch", lambda *args, **kwargs: fake_ctx)
    monkeypatch.setattr(train_module, "run_val_epoch", lambda *args, **kwargs: fake_ctx)
    monkeypatch.setattr(train_module, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module, "load_checkpoint", lambda *args, **kwargs: (1, 0.1, 0.9))
    monkeypatch.setattr(train_module, "export_pth", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module, "export_onnx", lambda *args, **kwargs: None)

    train_module.main()

    output = capsys.readouterr().out
    assert "Epoch 1/1" in output
    assert "Test |" in output
    assert fake_tracker.finished is True
    assert fake_tracker.metrics_calls[0] == (
        {"data/train_batches": 3, "data/val_batches": 2, "data/test_batches": 1},
        0,
    )
    assert fake_tracker.metrics_calls[1][0]["train/loss"] == 0.5
    assert fake_tracker.metrics_calls[2][0]["val/loss"] == 0.5
    assert fake_tracker.metrics_calls[2][0]["val/accuracy"] == 1.0
    assert fake_tracker.metrics_calls[-2][0]["test/accuracy"] == 1.0
    assert fake_tracker.metrics_calls[-1][0]["summary/best_val_loss"] == 0.5
