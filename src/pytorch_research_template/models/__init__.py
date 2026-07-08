"""Model plugins and registry factory."""

from pytorch_research_template.models.model_base import BuildModelFn, Model
from pytorch_research_template.models.model_factory import load_model, model_registry

__all__ = ["BuildModelFn", "Model", "load_model", "model_registry"]
