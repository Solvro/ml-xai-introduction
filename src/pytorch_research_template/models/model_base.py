"""Model plugin contracts."""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn
from omegaconf import DictConfig

Model = nn.Module
BuildModelFn = Callable[[DictConfig], Model]
