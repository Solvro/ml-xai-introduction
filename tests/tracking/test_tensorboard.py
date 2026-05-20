from __future__ import annotations

from dataclasses import dataclass, field

from omegaconf import OmegaConf

from ml_xai_introduction.tracking import tensorboard as tensorboard_tracker


@dataclass
class FakeSummaryWriter:
    log_dir: str
    scalars: list[tuple[str, object, int | None]] = field(default_factory=list)
    texts: list[tuple[str, str, int | None]] = field(default_factory=list)
    flushed: bool = False
    closed: bool = False

    def add_scalar(self, key: str, value: object, global_step: int | None = None) -> None:
        self.scalars.append((key, value, global_step))

    def add_text(self, key: str, value: str, global_step: int | None = None) -> None:
        self.texts.append((key, value, global_step))

    def flush(self) -> None:
        self.flushed = True

    def close(self) -> None:
        self.closed = True


def test_tensorboard_tracker_calls_expected_api(monkeypatch) -> None:
    monkeypatch.setattr(tensorboard_tracker, "_require_tensorboard_writer", lambda: FakeSummaryWriter)

    cfg = OmegaConf.create({
        "tensorboard": {"log_dir": "outputs/test/tensorboard"},
    })

    tracker = tensorboard_tracker.build_tracker(cfg)
    tracker.log_metrics({"loss": 0.15, "notes": {"status": "ok"}}, step=5)
    tracker.log_params({"batch_size": 64})
    tracker.log_artifact("artifact.txt", name="model")
    tracker.finish()

    writer = tracker._writer
    assert writer.log_dir == "outputs/test/tensorboard"
    assert writer.scalars == [("loss", 0.15, 5)]
    assert writer.texts[0][0] == "notes"
    assert writer.texts[1][0] == "params/config"
    assert writer.texts[2] == ("artifacts/model", "artifact.txt", None)
    assert writer.flushed is True
    assert writer.closed is True
