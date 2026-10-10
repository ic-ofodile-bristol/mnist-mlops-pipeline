import torch
from torch import nn, optim

from src.checkpoint import load_checkpoint, save_checkpoint
from src.engine import train_one_epoch
from src.model import SmallCNN


def test_train_and_resume_roundtrip(tmp_path, tiny_loader):
    model = SmallCNN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_one_epoch(model, tiny_loader, criterion, optimizer, "cpu")
    ckpt = tmp_path / "last.pt"
    save_checkpoint(ckpt, model, optimizer, epoch=0)

    # fresh model/optimizer, then restore
    model2 = SmallCNN()
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)
    last_epoch = load_checkpoint(ckpt, model2, optimizer2)

    assert last_epoch == 0
    assert ckpt.exists()