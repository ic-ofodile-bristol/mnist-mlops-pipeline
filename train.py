import argparse
from pathlib import Path

import torch
from torch import nn, optim

from src.config import load_config
from src.data import get_dataloaders
from src.engine import evaluate, train_one_epoch
from src.model import SmallCNN
from src.utils import get_device, set_seed


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    set_seed(cfg["seed"])
    device = get_device()

    train_loader, test_loader = get_dataloaders(batch_size=cfg["batch_size"])

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    for epoch in range(cfg["epochs"]):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        accuracy = evaluate(model, test_loader, device)
        print(f"epoch={epoch+1} loss={loss:.4f} accuracy={accuracy:.4f}")

    model_path = Path(cfg["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path} using {device}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    main(args.config)