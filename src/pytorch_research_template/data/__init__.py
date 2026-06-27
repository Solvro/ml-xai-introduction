"""Dataset plugins and registry factory."""

from pytorch_research_template.data.data_base import DataBundle, LoadDataFn
from pytorch_research_template.data.data_factory import dataset_registry, load_data

__all__ = ["DataBundle", "LoadDataFn", "dataset_registry", "load_data"]
