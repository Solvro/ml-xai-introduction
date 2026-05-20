"""Hydra entry point for loading a configured dataset."""

from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from ml_xai_introduction.data.data_factory import dataset
from ml_xai_introduction.tracking.factory import build_tracker


def _load_config() -> DictConfig:
    config_dir = Path(__file__).resolve().parents[2] / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name="train", overrides=sys.argv[1:])


def main() -> None:
    cfg = _load_config()
    print(OmegaConf.to_yaml(cfg, resolve=True))

    tracker = build_tracker(cfg.logging)

    try:
        tracker.log_params(OmegaConf.to_container(cfg, resolve=True))

        train_loader, test_loader = dataset(cfg)
        train_batch = next(iter(train_loader))
        images, labels = train_batch

        train_batches = len(train_loader)
        test_batches = len(test_loader)

        tracker.log_metrics({
            "data/train_batches": train_batches,
            "data/test_batches": test_batches,
            "data/image_height": int(images.shape[-2]),
            "data/image_width": int(images.shape[-1]),
        })

        print(f"Train batches: {train_batches}")
        print(f"Test batches: {test_batches}")
        print(f"First train batch images shape: {tuple(images.shape)}")
        print(f"First train batch labels shape: {tuple(labels.shape)}")
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
