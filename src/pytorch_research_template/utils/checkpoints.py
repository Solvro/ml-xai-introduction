"""Checkpoint save and load helpers."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer


def save_checkpoint(
    output_path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    loss: float,
    accuracy: float,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": loss,
        "val_accuracy": accuracy,
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(output_path: str | Path, model: nn.Module, optimizer: Optimizer) -> tuple[int, float, float]:
    path = Path(output_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {path}")

    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["val_loss"], checkpoint["val_accuracy"]
