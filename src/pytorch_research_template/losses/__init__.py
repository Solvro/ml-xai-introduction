"""Loss plugins and registry factory."""

from pytorch_research_template.losses.loss_base import BuildLossFn
from pytorch_research_template.losses.loss_factory import load_loss, loss_registry

__all__ = ["BuildLossFn", "load_loss", "loss_registry"]
