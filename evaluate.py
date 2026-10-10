import argparse

import torch

from src.config import load_config
from src.data import get_dataloaders
from src.engine import evaluate
from src.model import SmallCNN
from src.utils import get_device, set_seed, setup_logging


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    logger = setup_logging()

    set_seed(cfg["seed"])
    device = get_device()

    _, test_loader = get_dataloaders(batch_size=cfg["batch_size"])

    model = SmallCNN().to(device)
    model.load_state_dict(torch.load(cfg["model_path"], map_location=device))

    accuracy = evaluate(model, test_loader, device)
    logger.info("Evaluation accuracy=%.4f (model=%s)", accuracy, cfg["model_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    main(args.config)