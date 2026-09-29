"""Task heads for the window model."""
from __future__ import annotations

import torch
from torch import nn


def _mlp(d: int, out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, out))


class AnomalyHead(nn.Module):
    """MLP on [CLS] -> 1 logit per window."""

    def __init__(self, d: int):
        super().__init__()
        self.net = _mlp(d, 1)

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        return self.net(cls).squeeze(-1)


class SuspicionHead(nn.Module):
    """MLP on every line position -> 1 logit per line."""

    def __init__(self, d: int):
        super().__init__()
        self.net = _mlp(d, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


class TemplateHead(nn.Module):
    """Masked-line template prediction for unsupervised pretraining."""

    def __init__(self, d: int, n_templates: int):
        super().__init__()
        self.net = nn.Linear(d, n_templates)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)
