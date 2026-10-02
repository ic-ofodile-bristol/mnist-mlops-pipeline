"""Throwaway helper: export a few real MNIST test images to PNG for inference testing."""
from pathlib import Path

from torchvision import datasets

out = Path("samples")
out.mkdir(exist_ok=True)

test = datasets.MNIST("./data", train=False, download=True)  # PIL images, no transform
for i in range(5):
    image, label = test[i]
    image.save(out / f"sample_{i}_label{label}.png")
    print(f"saved samples/sample_{i}_label{label}.png (true label={label})")