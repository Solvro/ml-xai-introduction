"""Hydra entry point for the full training pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import torch
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_research_template.data.data_factory import load_data
from pytorch_research_template.metrics.metric_base import RunSummary
from pytorch_research_template.metrics.metric_logger import MetricLogger
from pytorch_research_template.metrics.metric_manager import MetricsManager
from pytorch_research_template.models.model_factory import load_model
from pytorch_research_template.tracking.tracking_factory import build_tracker
from pytorch_research_template.training.early_stopping import EarlyStopping
from pytorch_research_template.training.setup import setup_training
from pytorch_research_template.training.trainer import run_train_epoch, run_val_epoch
from pytorch_research_template.utils.checkpoints import load_checkpoint, save_checkpoint
from pytorch_research_template.utils.export import export_onnx, export_pth


def _load_config() -> DictConfig:
    config_dir = Path(__file__).resolve().parents[2] / "conf"
    job_name = Path(__file__).stem
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(
            config_name="train",
            overrides=[f"paths.job_name={job_name}", *sys.argv[1:]],
        )


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main() -> None:
    load_dotenv()
    cfg = _load_config()
    print(OmegaConf.to_yaml(cfg, resolve=True))

    device = _resolve_device()
    _set_seed(int(cfg.training.seed))

    tracker = build_tracker(cfg.logging)
    metrics_manager = MetricsManager(cfg.metrics)
    metric_logger = MetricLogger(tracker, cfg)
    summary = RunSummary()
    checkpoint_path = Path(cfg.paths.checkpoint_dir) / "best.pt"
    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)

    try:
        tracker.log_params(cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)))

        data = load_data(cfg)
        train_loader = data.train
        val_loader = data.val
        test_loader = data.test
        model = load_model(cfg).to(device)

        optimizer, criterion, scheduler = setup_training(
            model,
            cfg,
            steps_per_epoch=len(train_loader),
        )
        is_per_batch_scheduler = str(cfg.training.scheduler.name).lower() == "onecycle"

        metric_logger.log_run_start({
            "data/train_batches": len(train_loader),
            "data/val_batches": len(val_loader),
            "data/test_batches": len(test_loader),
        })

        es_cfg = cfg.training.early_stopping
        early_stopper: EarlyStopping | None = None
        if bool(es_cfg.enabled):
            early_stopper = EarlyStopping(
                patience=int(es_cfg.patience),
                min_delta=float(es_cfg.min_delta),
                monitor=str(es_cfg.monitor),
            )

        epochs = int(cfg.training.epochs)

        for epoch in range(epochs):
            train_ctx = run_train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scheduler=scheduler if is_per_batch_scheduler else None,
            )
            val_ctx = run_val_epoch(model, val_loader, criterion, device)
            train_metrics = metrics_manager.compute("train", train_ctx)
            val_metrics = metrics_manager.compute("validation", val_ctx)

            if scheduler is not None and not is_per_batch_scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    val_loss = float(val_metrics.get("loss", val_ctx.loss_sum / val_ctx.num_samples))
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            step = epoch + 1
            metric_logger.log_epoch("train", train_metrics, step=step)
            metric_logger.log_epoch("validation", val_metrics, step=step)
            metric_logger.log_extra({"learning_rate": current_lr}, step=step)
            metric_logger.print_epoch(epoch + 1, epochs, train_metrics, val_metrics, current_lr)

            val_loss = val_metrics.get("loss", float("inf"))
            val_accuracy = val_metrics.get("accuracy", 0.0)
            if val_loss < summary.best_val_loss:
                summary.best_val_loss = val_loss
                summary.best_val_accuracy = val_accuracy
                summary.best_epoch = epoch + 1
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    epoch + 1,
                    val_loss,
                    val_accuracy,
                )

            if early_stopper is not None:
                stop = early_stopper.step(val_metrics)
                if stop:
                    print(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(no improvement in {es_cfg.monitor} for {es_cfg.patience} epochs)"
                    )
                    break

        if checkpoint_path.exists():
            load_checkpoint(checkpoint_path, model, optimizer)

        test_ctx = run_val_epoch(model, test_loader, criterion, device)
        test_metrics = metrics_manager.compute("test", test_ctx)
        metric_logger.log_eval(test_metrics, step=epochs + 1)
        metric_logger.print_eval(test_metrics)

        metric_logger.log_summary({
            "best_epoch": summary.best_epoch,
            "best_val_loss": summary.best_val_loss,
            "best_val_accuracy": summary.best_val_accuracy,
        })

        export_path = Path(cfg.paths.output_dir) / f"{cfg.model.name}_best.pth"
        export_pth(model, export_path)
        if bool(cfg.training.export.onnx):
            onnx_path = Path(cfg.paths.output_dir) / f"{cfg.model.name}_best.onnx"
            export_onnx(model, onnx_path, device=str(device))
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
