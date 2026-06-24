"""Export trained models to disk."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def export_pth(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def export_onnx(
    model: nn.Module,
    path: str | Path,
    input_shape: tuple[int, ...] = (1, 1, 28, 28),
    device: str = "cpu",
) -> None:
    try:
        import onnx  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("ONNX export requires the 'onnx' package") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.zeros(input_shape, device=device)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
    )
