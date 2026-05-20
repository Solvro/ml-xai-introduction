from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from ml_xai_introduction.tracking import wandb as wandb_tracker


class FakeWandbRun:
    def __init__(self) -> None:
        self.logged_metrics: list[tuple[dict[str, object], int | None]] = []
        self.config_updates: list[dict[str, object]] = []
        self.artifacts: list[object] = []
        self.finished = False

    def log(self, metrics: dict[str, object], step: int | None = None) -> None:
        self.logged_metrics.append((metrics, step))

    @property
    def config(self):
        return SimpleNamespace(update=lambda params, allow_val_change: self.config_updates.append(params))

    def log_artifact(self, artifact) -> None:
        self.artifacts.append(artifact)

    def finish(self) -> None:
        self.finished = True


class FakeArtifact:
    def __init__(self, name: str, type: str) -> None:
        self.name = name
        self.type = type
        self.files: list[str] = []

    def add_file(self, path: str) -> None:
        self.files.append(path)


def test_wandb_tracker_calls_expected_api(monkeypatch, tmp_path: Path) -> None:
    fake_run = FakeWandbRun()
    fake_wandb = SimpleNamespace(init=lambda **kwargs: fake_run, Artifact=FakeArtifact)
    monkeypatch.setattr(wandb_tracker, "_require_wandb", lambda: fake_wandb)

    cfg = OmegaConf.create(
        {
            "project_name": "ml-xai-introduction",
            "experiment_name": "mnist-baseline",
            "wandb": {"mode": "offline", "entity": "demo", "tags": ["mnist"], "notes": "note"},
        }
    )

    tracker = wandb_tracker.build_tracker(cfg)
    tracker.log_metrics({"loss": 0.25}, step=3)
    tracker.log_params({"batch_size": 64})
    artifact_path = tmp_path / "model.pt"
    artifact_path.write_text("checkpoint")
    tracker.log_artifact(str(artifact_path), name="model")
    tracker.finish()

    assert fake_run.logged_metrics == [({"loss": 0.25}, 3)]
    assert fake_run.config_updates == [{"batch_size": 64}]
    assert fake_run.artifacts[0].name == "model"
    assert fake_run.artifacts[0].files == [str(artifact_path)]
    assert fake_run.finished is True