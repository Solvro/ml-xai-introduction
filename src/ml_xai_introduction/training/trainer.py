"""Training and validation epoch runners."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader

from ml_xai_introduction.metrics.metric_base import MetricContext


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> MetricContext:
    model.train()
    loss_sum = 0.0
    num_samples = 0
    all_predictions: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = x_batch.size(0)
        loss_sum += loss.item() * batch_size
        num_samples += batch_size
        all_predictions.append(logits.argmax(1))
        all_labels.append(y_batch)

    return MetricContext(
        predictions=torch.cat(all_predictions, 0),
        targets=torch.cat(all_labels, 0),
        loss_sum=loss_sum,
        num_samples=num_samples,
    )


@torch.no_grad()
def run_val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> MetricContext:
    model.eval()
    loss_sum = 0.0
    num_samples = 0
    all_predictions: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        batch_size = x_batch.size(0)
        loss_sum += loss.item() * batch_size
        num_samples += batch_size
        all_predictions.append(logits.argmax(1))
        all_labels.append(y_batch)

    return MetricContext(
        predictions=torch.cat(all_predictions, 0),
        targets=torch.cat(all_labels, 0),
        loss_sum=loss_sum,
        num_samples=num_samples,
    )
