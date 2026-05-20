## MNIST helper

Use `ml_xai_introduction.data.get_mnist_loaders()` to get MNIST train/test dataloaders. The first call downloads the dataset into `data/mnist` if it is missing.

```python
from ml_xai_introduction.data import get_mnist_loaders

train_loader, test_loader = get_mnist_loaders(batch_size=64)
```
