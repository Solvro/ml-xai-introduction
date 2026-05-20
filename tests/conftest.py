from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def tmp_mnist_root(tmp_path: Path) -> Path:
    return tmp_path / "mnist"


@pytest.fixture()
def tmp_fashion_mnist_root(tmp_path: Path) -> Path:
    return tmp_path / "fashion_mnist"