# ml-xai-introduction

Small PyTorch + Hydra playground for dataset loading and experiment tracking.

## Quick Start

Run the entrypoint with the default offline setup:

```bash
uv run python src/ml_xai_introduction/train.py
```

Override the dataset or tracker from the command line:

```bash
uv run python src/ml_xai_introduction/train.py data=mnist logging=wandb
uv run python src/ml_xai_introduction/train.py data=fashion_mnist logging=default
```

## How `train.py` Works

The entrypoint loads `conf/train.yaml` with Hydra, then:

1. Builds `cfg` from the selected config and CLI overrides.
2. Creates a tracker with `build_tracker(cfg.logging)`.
3. Logs the resolved config once at the start.
4. Loads the dataset through `dataset(cfg)`.
5. Logs basic dataset stats like number of batches and image shape.
6. Always calls `tracker.finish()` in a `finally` block.

If `logging=default`, the tracker stays offline/no-op.
If you choose `logging=wandb`, `logging=mlflow`, or `logging=tensorboard`, only that backend is initialized.

Minimal shape of the training entrypoint:

```python
cfg = _load_config()
tracker = build_tracker(cfg.logging)

try:
    tracker.log_params(OmegaConf.to_container(cfg, resolve=True))
    train_loader, test_loader = dataset(cfg)
    tracker.log_metrics({"data/train_batches": len(train_loader)})
finally:
    tracker.finish()
```

## Datasets

Datasets are loaded through the config-driven factory in `src/ml_xai_introduction/data/data_factory.py`.

- `data=mnist` → MNIST loaders
- `data=fashion_mnist` → FashionMNIST loaders

Example direct usage:

```python
from ml_xai_introduction.data.mnist import get_mnist_loaders
from ml_xai_introduction.data.fashion_mnist import get_fashion_mnist_loaders

train_loader, test_loader = get_mnist_loaders(batch_size=64)
fashion_train_loader, fashion_test_loader = get_fashion_mnist_loaders(batch_size=64)
```

Each helper checks whether the dataset already exists under `data/<name>` and downloads it automatically if needed.

## Logging

The tracking layer uses one small factory plus backend adapters in `src/ml_xai_introduction/tracking/`.

- `logging=default` → offline/no-op
- `logging=wandb` → Weights & Biases
- `logging=mlflow` → MLflow
- `logging=tensorboard` → TensorBoard

The public API is intentionally small: `log_params()`, `log_metrics()`, `log_artifact()`, and `finish()`.

## Quality And Coverage

The repository uses `pytest` for tests, `pytest-cov` for coverage reporting, and `ruff` for linting plus formatting.

Useful commands:

```bash
uv run pytest -q
uv run pytest --cov=src/ml_xai_introduction --cov-report=term-missing
uv run pytest --cov=src/ml_xai_introduction --cov-report=html
uv run ruff check .
uv run ruff format .
uv run pre-commit run --all-files
```

Current test coverage focuses on:

- dataset helpers for MNIST and FashionMNIST,
- the dataset factory dispatch layer,
- tracking manager dispatch,
- dynamic tracker factory resolution,
- backend adapters for `wandb`, `mlflow`, and `tensorboard`.
