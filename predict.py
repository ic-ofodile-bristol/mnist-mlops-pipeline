import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.config import load_config
from src.model import SmallCNN
from src.utils import get_device


def predict(image_path: str, config_path: str) -> int:
    cfg = load_config(config_path)
    device = get_device()

    model = SmallCNN().to(device)
    model.load_state_dict(torch.load(cfg["model_path"], map_location=device))
    model.eval()

    image = Image.open(image_path).convert("L").resize((28, 28))
    tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)  # (1, 1, 28, 28)

    with torch.no_grad():
        logits = model(tensor)
    return int(logits.argmax(dim=1).item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    print(f"Predicted digit: {predict(args.image, args.config)}")