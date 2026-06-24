"""Dataset plugins and registry factory."""

from ml_xai_introduction.data.data_base import DataBundle, LoadDataFn
from ml_xai_introduction.data.data_factory import dataset_registry, load_data

__all__ = ["DataBundle", "LoadDataFn", "dataset_registry", "load_data"]
