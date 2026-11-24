"""Feedforward neural network for 3-class review sentiment (PyTorch)."""

from __future__ import annotations

from torch import nn


class FeedforwardNet(nn.Module):
    """TF-IDF vector -> two hidden layers -> one score per class (Negative / Neutral / Positive)."""

    def __init__(self, input_dim: int, hidden1: int = 256, hidden2: int = 64, dropout: float = 0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden2, 3),  # raw scores (logits); softmax is applied later
        )

    def forward(self, x):
        return self.layers(x)
