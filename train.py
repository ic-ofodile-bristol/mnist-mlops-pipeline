import argparse
from pathlib import Path

import torch
from torch import nn, optim

from src.config import load_config
from src.data import get_dataloaders
from src.engine import evaluate, train_one_epoch
from src.model import SmallCNN
from src.utils import get_device, set_seed, setup_logging
from src.checkpoint import save_checkpoint, load_checkpoint

def main(config_path: str, resume: bool = False) -> None:
    cfg = load_config(config_path)
    logger = setup_logging()
    logger.info(f"Loaded")
    set_seed(cfg["seed"])
    device = get_device()

    train_loader, test_loader = get_dataloaders(batch_size=cfg["batch_size"])

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    
    ckpt_path = Path(cfg["checkpoint_dir"]) / "last.pt"

    start_epoch = 0
    if resume and ckpt_path.exists():
        start_epoch = load_checkpoint(ckpt_path, model, optimizer) + 1
        #logger.info(start_epoch)
        logger.info(f"Resumed from checkpoint at epoch {start_epoch}")
    try:
        best_accuracy = 0.0
        best_model_path = Path(cfg["model_path"]) 
        for epoch in range(start_epoch, cfg["epochs"]):
            loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            accuracy = evaluate(model, test_loader, device)
            logger.info(f"epoch={epoch+1} loss={loss:.4f} accuracy={accuracy:.4f}")
            print(epoch)
            save_checkpoint(ckpt_path, model, optimizer, epoch)
            print(epoch)
            if accuracy > best_accuracy:
                best_model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), best_model_path)
                logger.info("New best model (accuracy=%.4f) saved to %s", accuracy, best_model_path)
            logger.info(f"Saved checkpoint to {ckpt_path} at epoch {epoch+1}")
    except KeyboardInterrupt:
        logger.warning("Training interrupted")
        raise

    except Exception:
        logger.exception("Unexpected error occurred during training")
        raise
    model_path = Path(cfg["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model to {model_path} using {device}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    main(args.config, resume=args.resume)