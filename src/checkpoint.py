from pathlib import Path

import torch
from torch import nn


def save_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )

def load_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["epoch"]  # last completed epoch