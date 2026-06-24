"""Config-driven optimizer and scheduler setup."""

from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig
from torch import optim

from ml_xai_introduction.losses.loss_factory import load_loss


def build_optimizer(model: nn.Module, cfg_opt: DictConfig) -> optim.Optimizer:
    name = str(cfg_opt.name).lower()
    lr: float = float(cfg_opt.lr)
    weight_decay: float = float(cfg_opt.weight_decay)

    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=float(cfg_opt.momentum))
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(
    optimizer: optim.Optimizer,
    cfg_sched: DictConfig,
    epochs: int,
    steps_per_epoch: int | None = None,
) -> optim.lr_scheduler.LRScheduler | None:
    name = str(cfg_sched.name).lower()

    if name == "none":
        return None
    if name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg_sched.step_size), gamma=float(cfg_sched.gamma))
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(cfg_sched.T_max), eta_min=float(cfg_sched.eta_min)
        )
    if name == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=float(cfg_sched.factor),
            patience=int(cfg_sched.plateau_patience),
        )
    if name == "onecycle":
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch is required for OneCycleLR")
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(cfg_sched.max_lr),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
        )
    raise ValueError(f"Unknown scheduler: {name}")


def setup_training(
    model: nn.Module,
    cfg: DictConfig,
    steps_per_epoch: int | None = None,
) -> tuple[optim.Optimizer, nn.Module, optim.lr_scheduler.LRScheduler | None]:
    criterion = load_loss(cfg.loss, num_classes=int(cfg.data.num_classes))
    optimizer = build_optimizer(model, cfg.training.optimizer)
    scheduler = build_scheduler(
        optimizer,
        cfg.training.scheduler,
        epochs=int(cfg.training.epochs),
        steps_per_epoch=steps_per_epoch,
    )
    return optimizer, criterion, scheduler
