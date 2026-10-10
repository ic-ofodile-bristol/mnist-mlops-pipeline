import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import SmallCNN


@pytest.fixture
def model():
    return SmallCNN()


@pytest.fixture
def tiny_loader():
    # 8 fake MNIST-shaped images + labels -> fast, no download
    images = torch.randn(8, 1, 28, 28)
    labels = torch.randint(0, 10, (8,))
    return DataLoader(TensorDataset(images, labels), batch_size=4)