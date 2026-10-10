import torch
from torch import nn, optim

from src.engine import evaluate, train_one_epoch


def test_train_one_epoch_returns_loss(model, tiny_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss = train_one_epoch(model, tiny_loader, criterion, optimizer, "cpu")
    assert loss >= 0.0


def test_evaluate_returns_accuracy(model, tiny_loader):
    accuracy = evaluate(model, tiny_loader, "cpu")
    assert 0.0 <= accuracy <= 1.0