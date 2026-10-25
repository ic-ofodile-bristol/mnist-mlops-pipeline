import random

import numpy as np
import torch

import logging

from pathlib import Path

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(log_dir: str = "artifacts") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(log_dir) / "train.log"),
        ],
        force=True
    )
    return logging.getLogger("train")

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"