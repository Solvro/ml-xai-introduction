"""Hydra entry point for loading a configured dataset."""

from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from ml_xai_introduction.data.data_factory import dataset


def _load_config() -> DictConfig:
    config_dir = Path(__file__).resolve().parents[2] / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name="config", overrides=sys.argv[1:])


def main() -> None:
    cfg = _load_config()
    print(OmegaConf.to_yaml(cfg, resolve=True))

    train_loader, test_loader = dataset(cfg)
    train_batch = next(iter(train_loader))
    images, labels = train_batch

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"First train batch images shape: {tuple(images.shape)}")
    print(f"First train batch labels shape: {tuple(labels.shape)}")


if __name__ == "__main__":
    main()