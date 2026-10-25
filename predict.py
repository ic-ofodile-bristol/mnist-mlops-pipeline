import argparse

import mlflow
import mlflow.pytorch
import torch
from PIL import Image
from torchvision import transforms

from src.utils import get_device


def predict(image_path: str, alias: str = "champion") -> int:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    device = get_device()

    model = mlflow.pytorch.load_model(f"models:/mnist_best_model@{alias}")
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("L").resize((28, 28))
    tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
    return int(logits.argmax(dim=1).item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--alias", default="champion")
    args = parser.parse_args()
    print(f"Predicted digit: {predict(args.image, args.alias)}")