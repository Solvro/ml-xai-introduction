"""Loss plugins and registry factory."""

from ml_xai_introduction.losses.loss_base import BuildLossFn
from ml_xai_introduction.losses.loss_factory import load_loss, loss_registry

__all__ = ["BuildLossFn", "load_loss", "loss_registry"]
