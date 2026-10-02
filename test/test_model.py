import torch

from src.model import SmallCNN


def test_forward_output_shape():
    model = SmallCNN()
    batch = torch.randn(4, 1, 28, 28)
    out = model(batch)
    assert out.shape == (4, 10)