"""Loss plugin contracts."""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn
from omegaconf import DictConfig

BuildLossFn = Callable[[DictConfig, int], nn.Module]
