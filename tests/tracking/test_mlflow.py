from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from ml_xai_introduction.tracking import mlflow as mlflow_tracker


class FakeMLflow:
    def __init__(self) -> None:
        self.tracking_uris: list[str] = []
        self.experiments: list[str] = []
        self.runs: list[dict[str, object]] = []
        self.logged_metrics: list[tuple[dict[str, object], int | None]] = []
        self.logged_params: list[dict[str, object]] = []
        self.artifacts: list[tuple[str, str | None]] = []
        self.ended = False

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uris.append(uri)

    def set_experiment(self, name: str) -> None:
        self.experiments.append(name)

    def start_run(self, run_name: str, nested: bool):
        self.runs.append({"run_name": run_name, "nested": nested})
        return object()

    def log_metrics(self, metrics: dict[str, object], step: int | None = None) -> None:
        self.logged_metrics.append((metrics, step))

    def log_params(self, params: dict[str, object]) -> None:
        self.logged_params.append(params)

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        self.artifacts.append((path, artifact_path))

    def end_run(self) -> None:
        self.ended = True


def test_mlflow_tracker_calls_expected_api(monkeypatch, tmp_path: Path) -> None:
    fake_mlflow = FakeMLflow()
    monkeypatch.setattr(mlflow_tracker, "_require_mlflow", lambda: fake_mlflow)

    cfg = OmegaConf.create(
        {
            "project_name": "ml-xai-introduction",
            "experiment_name": "mnist-baseline",
            "mlflow": {"tracking_uri": "file:///tmp/mlruns", "nested": True, "tags": {}},
        }
    )

    tracker = mlflow_tracker.build_tracker(cfg)
    tracker.log_metrics({"accuracy": 0.9}, step=4)
    tracker.log_params({"nested": {"batch_size": 32}, "tags": ["mnist", "baseline"]})
    artifact_path = tmp_path / "metrics.json"
    artifact_path.write_text("{}")
    tracker.log_artifact(str(artifact_path), name="reports")
    tracker.finish()

    assert fake_mlflow.tracking_uris == ["file:///tmp/mlruns"]
    assert fake_mlflow.experiments == ["ml-xai-introduction"]
    assert fake_mlflow.runs == [{"run_name": "mnist-baseline", "nested": True}]
    assert fake_mlflow.logged_metrics == [({"accuracy": 0.9}, 4)]
    assert fake_mlflow.logged_params == [{"nested.batch_size": 32, "tags": '["mnist", "baseline"]'}]
    assert fake_mlflow.artifacts == [(str(artifact_path), "reports")]
    assert fake_mlflow.ended is True